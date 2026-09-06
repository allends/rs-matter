/*
 *
 *    Copyright (c) 2026 Project CHIP Authors
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

//! Matter Commissioner support.
//!
//! Building blocks for the **controller / commissioner** role — driving
//! a freshly-paired accessory through the standard commissioning
//! sequence and onto a fabric.
//!
//! # Scope and non-scope
//!
//! The types here orchestrate the *on-wire* commissioning flow only:
//! post-PASE invokes ([`Commissioner::commission`]) and post-AddNOC
//! CASE + `CommissioningComplete` ([`Commissioner::complete_via_case`]).
//!
//! Everything *off* the wire is the caller's responsibility:
//!
//!   - CA chain (RCAC, optional ICAC) generation — use [`cac::RcacGenerator`]
//!     / [`cac::IcacGenerator`]. In a real deployment the RCAC is
//!     minted once offline (typically on an HSM) and the ICAC at
//!     factory provisioning time. The commissioner only needs the ICAC
//!     private key + the RCAC and ICAC TLV certs at runtime.
//!   - Fabric install — i.e. [`crate::fabric::Fabrics::add`] with the
//!     controller's NOC, the RCAC/ICAC chain and an IPK. The caller
//!     does this once before running any commissioning, then reuses
//!     the resulting `fab_idx` for every device.
//!   - NodeID allocation — devices get NodeIDs the caller picks (whatever
//!     scheme they prefer: counter, hash, configuration). Same for the
//!     NOC's ASN.1 serial number; the caller can simply pass
//!     `serial == node_id` if they have no other constraint.
//!   - Persistence — everything the caller wants to survive a restart
//!     (ICAC private key, `fab_idx`, NOC-serial / next-NodeID counters,
//!     the fabric itself) is theirs to write and read back.
//!
//! See `tests/commissioning.rs` and `examples/src/bin/commissioner_tests.rs`
//! for a fully-worked example wiring of all of the above against a
//! single in-process fabric.
//!
//! # Phase split
//!
//! [`Commissioner::commission`] (over PASE) and
//! [`Commissioner::complete_via_case`] (over CASE) are split because
//! the rs-matter device responder requires `CommissioningComplete` to
//! arrive over a CASE session (Matter Core spec — and
//! enforced by `Failsafe::disarm` which calls `get_case_fab_idx`).

use core::num::NonZeroU8;

use crate::cert::gen::Validity;
use crate::crypto::{Crypto, RngCore, AEAD_CANON_KEY_LEN};
use crate::dm::clusters::gen_comm::{CommissioningErrorEnum, GeneralCommissioningClient};
use crate::dm::clusters::noc::{
    CertificateChainTypeEnum, NodeOperationalCertStatusEnum, OperationalCredentialsClient,
};
use crate::dm::endpoints::ROOT_ENDPOINT_ID;
use crate::dm::NodeId;
use crate::error::{Error, ErrorCode};
use crate::onboard::noc::NocGenerator;
use crate::tlv::{FromTLV, OctetStr, TLVElement};
use crate::transport::exchange::Exchange;
use crate::transport::network::Address;
use crate::transport::session::ATT_CHALLENGE_LEN;
use crate::transport::TransportPreference;
use crate::Matter;

pub mod cac;
pub mod noc;

/// NOCSRElements ([Matter Core spec]) is a struct with:
///   ctx(0) = `csr` (PKCS#10 CertificationRequest, DER-encoded)
///   ctx(1) = `CSRNonce` (32 bytes — must echo what we sent)
///   ctx(2..4) = vendor-reserved (ignored here)
const NOCSR_TAG_CSR: u8 = 1;
const NOCSR_TAG_NONCE: u8 = 2;

/// Knobs for [`Commissioner::commission`].
///
/// Per-device material (NodeID, validity) is passed as separate
/// arguments to [`Commissioner::commission`] — it's expected to vary
/// every call. This struct carries only the per-flow tunables.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct CommissionOptions {
    /// `ExpiryLengthSeconds` for `ArmFailSafe`. The whole commissioning
    /// flow must complete before this expires, otherwise the device
    /// rolls back any partial state.
    pub fail_safe_secs: u16,
    /// Allow [`Commissioner::commission`] to accept attestation evidence
    /// without verification. Suitable only for test devices.
    ///
    /// Production callers should leave this `false` and use
    /// [`Commissioner::commission_with_attestation`] instead.
    pub allow_test_attestation: bool,
}

impl CommissionOptions {
    pub const fn new() -> Self {
        Self {
            fail_safe_secs: 60,
            allow_test_attestation: false,
        }
    }
}

impl Default for CommissionOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Device attestation material borrowed directly from commissioning responses.
///
/// Each value is valid only for the duration of the verifier callback. Values
/// arrive in variant order; staging lets the commissioner remain zero-copy
/// despite the transport's single receive buffer.
pub enum AttestationEvidence<'a> {
    /// Device Attestation Certificate bytes.
    Dac(&'a [u8]),
    /// Product Attestation Intermediate certificate bytes.
    Pai(&'a [u8]),
    /// Attestation response bytes and the inputs needed to verify its signature.
    Attestation {
        elements: &'a [u8],
        signature: &'a [u8],
        nonce: &'a [u8; 32],
        challenge: &'a [u8; ATT_CHALLENGE_LEN],
    },
}

/// Opaque evidence that phase 2 completed over an authenticated CASE session.
///
/// Values are captured from the library-owned session actually used to receive a
/// successful `CommissioningComplete` response. Private fields make it impossible
/// for applications to manufacture commissioning success from requested identity.
///
/// ```compile_fail
/// use core::num::NonZeroU8;
/// use rs_matter::onboard::CommissioningCaseProof;
///
/// let _forged = CommissioningCaseProof {
///     controller_fabric_index: NonZeroU8::new(1).unwrap(),
///     commissioned_node_id: 1,
///     session_id: 1,
///     local_session_id: 1,
///     peer_session_id: 1,
/// };
/// ```
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct CommissioningCaseProof {
    controller_fabric_index: NonZeroU8,
    commissioned_node_id: NodeId,
    session_id: u32,
    local_session_id: u16,
    peer_session_id: u16,
}

impl CommissioningCaseProof {
    /// Controller-side fabric used by the authenticated CASE session.
    pub const fn controller_fabric_index(&self) -> NonZeroU8 {
        self.controller_fabric_index
    }

    /// Operational node authenticated by the CASE session.
    pub const fn commissioned_node_id(&self) -> NodeId {
        self.commissioned_node_id
    }

    /// Library-internal unique identity of the CASE session used for completion.
    pub const fn session_id(&self) -> u32 {
        self.session_id
    }

    /// Local protocol session identifier negotiated by CASE.
    pub const fn local_session_id(&self) -> u16 {
        self.local_session_id
    }

    /// Peer protocol session identifier negotiated by CASE.
    pub const fn peer_session_id(&self) -> u16 {
        self.peer_session_id
    }
}

/// Opaque evidence that `OperationalCredentials::RemoveFabric` completed
/// successfully over the expected authenticated CASE session.
///
/// The proof is issued only by [`remove_fabric_via_case`] after the peer returns
/// an `NOCResponse` with `OK` and echoes the exact requested fabric index. The
/// CASE session identity is checked both before the invoke and after its
/// acknowledgement completes.
///
/// ```compile_fail
/// use core::num::NonZeroU8;
/// use rs_matter::onboard::RemoveFabricCaseProof;
///
/// let _forged = RemoveFabricCaseProof {
///     controller_fabric_index: NonZeroU8::new(1).unwrap(),
///     peer_node_id: 1,
///     removed_fabric_index: NonZeroU8::new(1).unwrap(),
///     session_id: 1,
///     local_session_id: 1,
///     peer_session_id: 1,
/// };
/// ```
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct RemoveFabricCaseProof {
    controller_fabric_index: NonZeroU8,
    peer_node_id: NodeId,
    removed_fabric_index: NonZeroU8,
    session_id: u32,
    local_session_id: u16,
    peer_session_id: u16,
}

impl RemoveFabricCaseProof {
    /// Controller-side fabric used to authenticate the CASE session.
    pub const fn controller_fabric_index(&self) -> NonZeroU8 {
        self.controller_fabric_index
    }

    /// Operational node authenticated by the CASE session.
    pub const fn peer_node_id(&self) -> NodeId {
        self.peer_node_id
    }

    /// Peer-side fabric slot that the successful command removed.
    pub const fn removed_fabric_index(&self) -> NonZeroU8 {
        self.removed_fabric_index
    }

    /// Library-internal unique identity of the CASE session used by the invoke.
    pub const fn session_id(&self) -> u32 {
        self.session_id
    }

    /// Local protocol session identifier negotiated by CASE.
    pub const fn local_session_id(&self) -> u16 {
        self.local_session_id
    }

    /// Peer protocol session identifier negotiated by CASE.
    pub const fn peer_session_id(&self) -> u16 {
        self.peer_session_id
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct CaseSessionIdentity {
    session_id: u32,
    local_session_id: u16,
    peer_session_id: u16,
}

fn case_session_identity(
    matter: &Matter<'_>,
    session_id: u32,
    controller_fabric_index: NonZeroU8,
    peer_node_id: NodeId,
) -> Result<CaseSessionIdentity, Error> {
    matter.with_state(|state| {
        let session = state.sessions.get(session_id).ok_or(ErrorCode::NoSession)?;

        if !session.is_for_node(controller_fabric_index, peer_node_id) {
            return Err(ErrorCode::InvalidState.into());
        }

        Ok(CaseSessionIdentity {
            session_id: session.id(),
            local_session_id: session.get_local_sess_id(),
            peer_session_id: session.get_peer_sess_id(),
        })
    })
}

/// Invoke `OperationalCredentials::RemoveFabric` over authenticated CASE and
/// return library-issued evidence of the exact successful operation.
///
/// This removes only the requested fabric on the peer. It does not mutate the
/// controller's local fabric table and never falls back to factory reset.
pub async fn remove_fabric_via_case<C: Crypto>(
    matter: &Matter<'_>,
    crypto: &C,
    controller_fabric_index: NonZeroU8,
    peer_node_id: NodeId,
    peer_fabric_index: NonZeroU8,
) -> Result<RemoveFabricCaseProof, Error> {
    remove_fabric_via_case_with_transport(
        matter,
        crypto,
        controller_fabric_index,
        peer_node_id,
        peer_fabric_index,
        TransportPreference::default(),
    )
    .await
}

/// Transport-selecting variant of [`remove_fabric_via_case`].
pub async fn remove_fabric_via_case_with_transport<C: Crypto>(
    matter: &Matter<'_>,
    crypto: &C,
    controller_fabric_index: NonZeroU8,
    peer_node_id: NodeId,
    peer_fabric_index: NonZeroU8,
    transport: TransportPreference,
) -> Result<RemoveFabricCaseProof, Error> {
    let exchange = Exchange::initiate_with_transport(
        matter,
        crypto,
        controller_fabric_index,
        peer_node_id,
        transport,
    )
    .await?;

    let exchange_id = exchange.id();
    let identity = case_session_identity(
        matter,
        exchange_id.session_id(),
        controller_fabric_index,
        peer_node_id,
    )?;

    let handle = exchange
        .operational_credentials()
        .remove_fabric(ROOT_ENDPOINT_ID, |req| {
            req.fabric_index(peer_fabric_index.get())?.end()
        })
        .await?;

    let (status, response_fabric_index) = {
        let response = handle.response()?;
        (response.status_code()?, response.fabric_index()?)
    };

    handle.complete().await?;

    if status != NodeOperationalCertStatusEnum::OK {
        return Err(ErrorCode::Failure.into());
    }

    if response_fabric_index != Some(peer_fabric_index.get()) {
        return Err(ErrorCode::InvalidData.into());
    }

    let completed_identity = case_session_identity(
        matter,
        identity.session_id,
        controller_fabric_index,
        peer_node_id,
    )?;

    if completed_identity != identity {
        return Err(ErrorCode::InvalidState.into());
    }

    Ok(RemoveFabricCaseProof {
        controller_fabric_index,
        peer_node_id,
        removed_fabric_index: peer_fabric_index,
        session_id: identity.session_id,
        local_session_id: identity.local_session_id,
        peer_session_id: identity.peer_session_id,
    })
}

/// What [`Commissioner::commission`] returns on success.
///
/// Also the handoff between phase 1 (`commission`) and phase 2
/// ([`Commissioner::complete_via_case`]).
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct CommissionResult {
    /// Fabric slot the **device** assigned to us. Needed for subsequent
    /// `UpdateNOC` / `RemoveFabric` / `UpdateFabricLabel` invocations.
    /// (Independent of whatever local fabric index the **controller**
    /// recorded for the same fabric — see [`Commissioner::fab_idx`].)
    ///
    /// `NonZeroU8` because the Matter Core spec reserves `fabric_index=0`
    /// for "no fabric" / PASE — a successful `NOCResponse` carrying a
    /// device-side fabric slot is, by definition, non-zero.
    pub fabric_index: NonZeroU8,
    /// Echo of the NodeID the caller supplied to
    /// [`Commissioner::commission`] — kept here so the same struct can
    /// be threaded into [`Commissioner::complete_via_case`] without the
    /// caller having to plumb it separately.
    pub device_node_id: NodeId,
}

/// Stateful commissioner.
///
/// Holds the references needed for the whole flow so individual steps
/// don't have to take them. `&mut NocGenerator` because each
/// `commission()` call mutably borrows the generator's scratch buffer
/// to write the device NOC into; `&mut [u8] buf` is a caller-owned
/// scratch slice used to stage the fabric's RCAC and ICAC bytes
/// across the on-wire async calls (the fabric record itself can only
/// be borrowed inside [`Matter::with_state`], which doesn't compose
/// with `await`).
///
/// **The controller's fabric is expected to already be in
/// `matter.state.fabrics`** at the given `fab_idx` — the caller installs
/// it once via [`crate::fabric::Fabrics::add`] before constructing any
/// commissioner. A single `Commissioner` instance can then be reused to
/// commission any number of devices onto that fabric.
pub struct Commissioner<'a, C: Crypto> {
    matter: &'a Matter<'a>,
    crypto: C,
    fab_idx: NonZeroU8,
    noc_generator: &'a mut NocGenerator<'a>,
    buf: &'a mut [u8],
    case_attempts: u8,
    transport: TransportPreference,
}

impl<'a, C: Crypto> Commissioner<'a, C> {
    /// Create a commissioner bound to a Matter stack, crypto backend,
    /// an already-installed fabric (`fab_idx`), an already-constructed
    /// NOC generator that signs against the chain stored on that
    /// fabric, and a scratch buffer.
    ///
    /// `buf` is used to copy the fabric's RCAC and (optionally) ICAC
    /// bytes out of the locked fabric table so they can be passed to
    /// the asynchronous `AddTrustedRootCertificate` / `AddNOC` invokes.
    /// It must be at least [`crate::cert::MAX_CERT_TLV_LEN`] bytes; the
    /// commissioner sequences the two transfers (RCAC first, then ICAC
    /// re-uses the same slot) so a single-cert worth of memory is
    /// enough.
    pub const fn new(
        matter: &'a Matter<'a>,
        crypto: C,
        fab_idx: NonZeroU8,
        noc_generator: &'a mut NocGenerator<'a>,
        buf: &'a mut [u8],
    ) -> Self {
        Self {
            matter,
            crypto,
            fab_idx,
            noc_generator,
            buf,
            case_attempts: Self::DEFAULT_CASE_ATTEMPTS,
            transport: TransportPreference::Mrp,
        }
    }

    /// How many times [`Self::complete_via_case`] drives the whole
    /// phase-2 sequence before giving up. See [`Self::with_case_attempts`].
    pub const DEFAULT_CASE_ATTEMPTS: u8 = 3;

    /// Override how many times phase 2 is attempted (default
    /// [`Self::DEFAULT_CASE_ATTEMPTS`]).
    ///
    /// Phase 2 runs immediately after the device has been told to join its
    /// operational network, so the first attempt regularly races the device: it
    /// may not have associated yet, or not yet be announcing over mDNS, and the
    /// resolve simply finds nothing. Retrying the whole sequence - discovery
    /// included - is what turns that race into a delay instead of a failure.
    ///
    /// This is deliberately scoped to commissioning. The CHIP SDK does the same:
    /// `FindOrEstablishSession` takes an `attemptCount` that defaults to 1, and
    /// the commissioner is the only caller that raises it (to 3).
    ///
    /// A value of 0 is treated as 1 (a single attempt).
    pub const fn with_case_attempts(mut self, attempts: u8) -> Self {
        self.case_attempts = attempts;
        self
    }

    /// Override the transport phase 2 establishes its CASE session over
    /// (default [`TransportPreference::Mrp`], i.e. UDP).
    ///
    /// Commissioning is a small-payload exchange, so UDP is the right default and
    /// what the CHIP SDK's commissioner always uses. The knob exists for devices
    /// that run no UDP transport at all - rs-matter permits a TCP-only node, and
    /// such a device cannot be finalised over UDP.
    pub const fn with_transport(mut self, transport: TransportPreference) -> Self {
        self.transport = transport;
        self
    }

    /// Index of the controller's fabric in `matter.state.fabrics`. The
    /// caller picked this when installing the fabric; the commissioner
    /// simply propagates it (e.g. into [`CommissionResult`] callers
    /// build on top).
    pub const fn fab_idx(&self) -> NonZeroU8 {
        self.fab_idx
    }

    /// Phase 1 — drive `ArmFailSafe` through `AddNOC` over PASE.
    ///
    /// Pre-condition: PASE handshake against the device has completed
    /// successfully on `matter`'s transport. The function locates that
    /// PASE session by the `(fab=0, peer=0, secure=true)` lookup tuple
    /// every step uses — it implicitly assumes a single in-flight PASE
    /// session, which is the case in practice for a controller driving
    /// one device at a time.
    ///
    /// `device_node_id` is the NodeID the caller wishes to assign to
    /// the device on the controller's fabric. `validity` is the NOC's
    /// validity window — typically [`crate::cert::gen::VALID_FOREVER`]
    /// for long-lived deployments, or a bounded window for short-lived
    /// re-issuance. The NOC's ASN.1 serial number is derived from the
    /// NodeID (see [`NocGenerator::generate`]).
    ///
    /// On success the device has accepted our RCAC + NOC and assigned
    /// us a [`CommissionResult::fabric_index`], but its fail-safe is
    /// still armed and PASE is still live. Phase 2
    /// ([`Self::complete_via_case`]) finalises commissioning over
    /// CASE; if the caller doesn't run it before the fail-safe expires
    /// the device rolls back.
    pub async fn commission(
        &mut self,
        peer_addr: Address,
        passcode: u32,
        opts: &CommissionOptions,
        device_node_id: NodeId,
        validity: Validity,
    ) -> Result<CommissionResult, Error> {
        if !opts.allow_test_attestation {
            return Err(ErrorCode::Failure.into());
        }

        self.commission_with_attestation(
            peer_addr,
            passcode,
            opts,
            device_node_id,
            validity,
            |_| Ok(()),
        )
        .await
    }

    /// Phase 1 commissioning with caller-supplied device attestation verification.
    ///
    /// `verify` is called for DAC, PAI, then the attestation response, while each
    /// response buffer is live and before it is completed. Any callback error is
    /// returned before CSRRequest, AddTrustedRootCertificate, or AddNOC is sent.
    pub async fn commission_with_attestation<F>(
        &mut self,
        peer_addr: Address,
        passcode: u32,
        opts: &CommissionOptions,
        device_node_id: NodeId,
        validity: Validity,
        verify: F,
    ) -> Result<CommissionResult, Error>
    where
        F: for<'e> FnMut(AttestationEvidence<'e>) -> Result<(), Error>,
    {
        self.arm_fail_safe(peer_addr, passcode, opts.fail_safe_secs)
            .await?;

        let mut attestation_nonce = [0u8; 32];
        self.crypto.rand()?.fill_bytes(&mut attestation_nonce);
        self.verify_device_attestation(peer_addr, passcode, &attestation_nonce, verify)
            .await?;

        // CSRRequest: random 32B nonce, then validate the device's echo
        // and mint the operational NOC in the same scope where the CSR
        // is borrowed from the response RX buffer — no per-call staging
        // copy of the (up to ~400-byte) DER blob on our stack.
        let mut csr_nonce = [0u8; 32];
        self.crypto.rand()?.fill_bytes(&mut csr_nonce);

        // Field-projection borrows so the closure passed to
        // `csr_request` (and the buf-staged AddTrustedRoot / AddNOC
        // calls below) don't conflict with a `&self` borrow.
        let matter = self.matter;
        let crypto = &self.crypto;
        let fab_idx = self.fab_idx;
        let noc_generator = &mut *self.noc_generator;
        let buf = &mut *self.buf;

        // Sign the device NOC. The returned slice lives in
        // `noc_generator.buf` — independent of `buf`, so we can use
        // both side-by-side below.
        let noc = Self::csr_request(matter, crypto, peer_addr, passcode, &csr_nonce, |csr_der| {
            noc_generator.generate(crypto, csr_der, device_node_id, &[], validity)
        })
        .await?;

        // Stage the RCAC in `buf`, send `AddTrustedRootCertificate`,
        // then re-use the same slot for the ICAC + grab IPK and admin
        // scalars on the way. Two `with_state` passes (cheap — mutex
        // + table lookup) keep the staging buffer to a single
        // `MAX_CERT_TLV_LEN` slot. IPK is a 16-byte fixed-size stack
        // array — trivial.
        let rcac_len = matter.with_state(|state| {
            let fabric = state.fabrics.fabric(fab_idx)?;
            let rcac = fabric.root_ca();
            if rcac.len() > buf.len() {
                return Err(Error::from(ErrorCode::BufferTooSmall));
            }
            buf[..rcac.len()].copy_from_slice(rcac);
            Ok::<_, Error>(rcac.len())
        })?;
        Self::add_trusted_root_certificate(matter, crypto, peer_addr, passcode, &buf[..rcac_len])
            .await?;

        // IPK as sent on the wire is the **epoch key** (the raw
        // 16-byte input to the group-key derivation), not the
        // per-fabric derived `op_key`. `KeySet` stores both;
        // `.epoch_key()` is the right one.
        let mut ipk_bytes = [0u8; AEAD_CANON_KEY_LEN];
        let (icac_len, admin_node_id, admin_vendor_id) = matter.with_state(|state| {
            let fabric = state.fabrics.fabric(fab_idx)?;
            let icac = fabric.icac();
            if icac.len() > buf.len() {
                return Err(Error::from(ErrorCode::BufferTooSmall));
            }
            buf[..icac.len()].copy_from_slice(icac);
            ipk_bytes.copy_from_slice(fabric.ipk().epoch_key().access());
            Ok::<_, Error>((icac.len(), fabric.node_id(), fabric.vendor_id()))
        })?;

        // AddNOC. `&buf[..icac_len]` is empty for RCAC-direct fabrics
        // (the codegen builder skips the field entirely); non-empty ⇒
        // the full `[RCAC, ICAC, NOC]` chain is shipped.
        let fabric_index = Self::add_noc(
            matter,
            crypto,
            peer_addr,
            passcode,
            noc,
            &buf[..icac_len],
            &ipk_bytes,
            admin_node_id,
            admin_vendor_id,
        )
        .await?;

        Ok(CommissionResult {
            fabric_index,
            device_node_id,
        })
    }

    /// Phase 2 — establish CASE against the device's freshly-installed
    /// operational identity and invoke `CommissioningComplete` over it.
    ///
    /// The device's operational endpoint is discovered via `_matter._tcp` mDNS
    /// from `(fabric, device_node_id)`.
    ///
    /// The sequence is attempted up to [`Self::with_case_attempts`] times: it
    /// runs immediately after the device was told to join its network, so losing
    /// the first race against the device coming online is routine.
    pub async fn complete_via_case(
        &mut self,
        phase1: &CommissionResult,
    ) -> Result<CommissioningCaseProof, Error> {
        let fab_idx = self.fab_idx;

        let mut last_err = None;

        for attempt in 0..self.case_attempts.max(1) {
            match self
                .commissioning_complete(fab_idx, phase1.device_node_id)
                .await
            {
                Ok(proof) => return Ok(proof),
                Err(err) if Self::phase2_worth_retrying(&err) => {
                    warn!(
                        "Commissioning phase 2 attempt {} failed: {}; retrying",
                        attempt + 1,
                        err
                    );
                    last_err = Some(err);
                }
                Err(err) => return Err(err),
            }
        }

        Err(last_err.unwrap_or_else(|| ErrorCode::Failure.into()))
    }

    /// `GeneralCommissioning::ArmFailSafe(expiry, breadcrumb=0)`.
    pub(crate) async fn arm_fail_safe(
        &self,
        peer_addr: Address,
        passcode: u32,
        expiry_seconds: u16,
    ) -> Result<(), Error> {
        let exchange =
            Exchange::initiate_pase(self.matter, &self.crypto, peer_addr, passcode).await?;

        let handle = exchange
            .general_commissioning()
            .arm_fail_safe(ROOT_ENDPOINT_ID, |req| {
                req.expiry_length_seconds(expiry_seconds)?
                    .breadcrumb(0)?
                    .end()
            })
            .await?;

        let code = handle.response()?.error_code()?;

        handle.complete().await?;

        if code != CommissioningErrorEnum::OK {
            return Err(ErrorCode::Failure.into());
        }

        Ok(())
    }

    /// `OperationalCredentials::CSRRequest(nonce)` — hands the
    /// DER-encoded PKCS#10 CSR pulled out of the NOCSRElements payload
    /// to `use_csr` (with the nonce echo already validated).
    ///
    /// The CSR slice handed to the closure is borrowed *directly* from
    /// the response RX buffer — no per-call staging copy. The buffer
    /// stays alive for the duration of `use_csr`; the trailing
    /// `StatusResponse(Success)` ACK is sent after it returns.
    ///
    /// Static-style (no `&self` receiver) so the caller can pass a
    /// closure that re-borrows other `Commissioner` fields (e.g.
    /// `noc_generator`, `crypto`) without conflicting with a `&self`
    /// borrow on this method.
    pub(crate) async fn csr_request<'m, F, R>(
        matter: &'m Matter<'m>,
        crypto: &C,
        peer_addr: Address,
        passcode: u32,
        csr_nonce: &[u8; 32],
        use_csr: F,
    ) -> Result<R, Error>
    where
        F: FnOnce(&[u8]) -> Result<R, Error>,
    {
        let exchange = Exchange::initiate_pase(matter, crypto, peer_addr, passcode).await?;

        let handle = exchange
            .operational_credentials()
            .csr_request(ROOT_ENDPOINT_ID, |req| {
                req.csr_nonce(OctetStr::new(csr_nonce))?
                    .is_for_update_noc(None)?
                    .end()
            })
            .await?;

        let result = {
            let resp = handle.response()?;
            let nocsr_bytes = resp.nocsr_elements()?;

            // NOCSRElements is itself TLV — its `csr` and `CSRNonce`
            // fields live at ctx(1) and ctx(2) of an anonymous struct.
            let root = TLVElement::new(nocsr_bytes.0).structure()?;
            let csr_tlv = OctetStr::from_tlv(&root.ctx(NOCSR_TAG_CSR)?)?;
            let nonce_echo = OctetStr::from_tlv(&root.ctx(NOCSR_TAG_NONCE)?)?;

            if nonce_echo.0 != csr_nonce {
                // Replay / freshness failure — abort before minting a NOC.
                return Err(ErrorCode::Failure.into());
            }

            use_csr(csr_tlv.0)?
        };

        handle.complete().await?;

        Ok(result)
    }

    /// `OperationalCredentials::AddTrustedRootCertificate(rcac_tlv)`.
    ///
    /// Static-style for the same reason as [`Self::csr_request`] — the
    /// caller in [`Self::commission`] is holding a `&mut self.noc_generator`
    /// projection borrow when it invokes this.
    pub(crate) async fn add_trusted_root_certificate<'m>(
        matter: &'m Matter<'m>,
        crypto: &C,
        peer_addr: Address,
        passcode: u32,
        rcac_tlv: &[u8],
    ) -> Result<(), Error> {
        let exchange = Exchange::initiate_pase(matter, crypto, peer_addr, passcode).await?;

        exchange
            .operational_credentials()
            .add_trusted_root_certificate(ROOT_ENDPOINT_ID, |req| {
                req.root_ca_certificate(OctetStr::new(rcac_tlv))?.end()
            })
            .await
    }

    /// `OperationalCredentials::AddNOC(noc, icac?, ipk, admin_subject,
    /// admin_vendor_id)` — returns the FabricIndex the device assigned.
    ///
    /// Pass an empty `icac` slice when the controller signs NOCs
    /// directly off the RCAC (no ICAC tier).
    ///
    /// Static-style: see [`Self::add_trusted_root_certificate`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn add_noc<'m>(
        matter: &'m Matter<'m>,
        crypto: &C,
        peer_addr: Address,
        passcode: u32,
        noc: &[u8],
        icac: &[u8],
        ipk: &[u8],
        admin_case_subject: u64,
        admin_vendor_id: u16,
    ) -> Result<NonZeroU8, Error> {
        let exchange = Exchange::initiate_pase(matter, crypto, peer_addr, passcode).await?;

        let handle = exchange
            .operational_credentials()
            .add_noc(ROOT_ENDPOINT_ID, |req| {
                req.noc_value(OctetStr::new(noc))?
                    .icac_value(if icac.is_empty() {
                        None
                    } else {
                        Some(OctetStr::new(icac))
                    })?
                    .ipk_value(OctetStr::new(ipk))?
                    .case_admin_subject(admin_case_subject)?
                    .admin_vendor_id(admin_vendor_id)?
                    .end()
            })
            .await?;

        let (status, fabric_index) = {
            let resp = handle.response()?;
            (resp.status_code()?, resp.fabric_index()?)
        };

        handle.complete().await?;

        if status != NodeOperationalCertStatusEnum::OK {
            return Err(ErrorCode::Failure.into());
        }

        // Spec reserves `fabric_index=0` for PASE / no-fabric; the
        // device must assign a non-zero slot on a successful `AddNOC`.
        // A missing field or a zero value is a peer-side bug — surface
        // it as `InvalidData` rather than silently widening.
        fabric_index
            .and_then(NonZeroU8::new)
            .ok_or_else(|| ErrorCode::InvalidData.into())
    }

    /// Whether a failed phase-2 attempt is worth driving the whole sequence
    /// again.
    fn phase2_worth_retrying(err: &Error) -> bool {
        err.is_peer_unresponsive()
            || matches!(
                err.code(),
                ErrorCode::NotFound | ErrorCode::NoSession | ErrorCode::Busy
            )
    }

    pub(crate) async fn commissioning_complete(
        &self,
        fab_idx: NonZeroU8,
        peer_node_id: NodeId,
    ) -> Result<CommissioningCaseProof, Error> {
        let exchange = Exchange::initiate_with_transport(
            self.matter,
            &self.crypto,
            fab_idx,
            peer_node_id,
            self.transport,
        )
        .await?;

        let (session_id, local_session_id, peer_session_id) = exchange.with_state(|state| {
            let session = exchange.id().session(&mut state.sessions);
            if !session.is_for_node(fab_idx, peer_node_id) {
                return Err(ErrorCode::InvalidState.into());
            }
            Ok((
                session.id(),
                session.get_local_sess_id(),
                session.get_peer_sess_id(),
            ))
        })?;

        let handle = exchange
            .general_commissioning()
            .commissioning_complete(ROOT_ENDPOINT_ID)
            .await?;

        let code = handle.response()?.error_code()?;

        handle.complete().await?;

        if code != CommissioningErrorEnum::OK {
            return Err(ErrorCode::Failure.into());
        }

        Ok(CommissioningCaseProof {
            controller_fabric_index: fab_idx,
            commissioned_node_id: peer_node_id,
            session_id,
            local_session_id,
            peer_session_id,
        })
    }

    async fn verify_device_attestation<F>(
        &self,
        peer_addr: Address,
        passcode: u32,
        nonce: &[u8; 32],
        mut verify: F,
    ) -> Result<(), Error>
    where
        F: for<'e> FnMut(AttestationEvidence<'e>) -> Result<(), Error>,
    {
        let dac_exchange =
            Exchange::initiate_pase(self.matter, &self.crypto, peer_addr, passcode).await?;
        let mut challenge = [0; ATT_CHALLENGE_LEN];
        dac_exchange.with_state(|state| {
            let session = dac_exchange.id().session(&mut state.sessions);
            challenge.copy_from_slice(
                session
                    .get_att_challenge()
                    .ok_or(ErrorCode::InvalidState)?
                    .access(),
            );
            Ok(())
        })?;
        let dac_handle = dac_exchange
            .operational_credentials()
            .certificate_chain_request(ROOT_ENDPOINT_ID, |req| {
                req.certificate_type(CertificateChainTypeEnum::DACCertificate)?
                    .end()
            })
            .await?;
        let verification = {
            let response = dac_handle.response()?;
            verify(AttestationEvidence::Dac(response.certificate()?.0))
        };
        dac_handle.complete().await?;
        verification?;

        let pai_handle = Exchange::initiate_pase(self.matter, &self.crypto, peer_addr, passcode)
            .await?
            .operational_credentials()
            .certificate_chain_request(ROOT_ENDPOINT_ID, |req| {
                req.certificate_type(CertificateChainTypeEnum::PAICertificate)?
                    .end()
            })
            .await?;
        let verification = {
            let response = pai_handle.response()?;
            verify(AttestationEvidence::Pai(response.certificate()?.0))
        };
        pai_handle.complete().await?;
        verification?;

        let attestation_handle =
            Exchange::initiate_pase(self.matter, &self.crypto, peer_addr, passcode)
                .await?
                .operational_credentials()
                .attestation_request(ROOT_ENDPOINT_ID, |req| {
                    req.attestation_nonce(OctetStr::new(nonce))?.end()
                })
                .await?;
        let verification = {
            let response = attestation_handle.response()?;
            verify(AttestationEvidence::Attestation {
                elements: response.attestation_elements()?.0,
                signature: response.attestation_signature()?.0,
                nonce,
                challenge: &challenge,
            })
        };
        attestation_handle.complete().await?;

        verification
    }
}
