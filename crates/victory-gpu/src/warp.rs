//! The WARP device that the GPU layer's D3D12 behaviour tests (queue waits,
//! submissions in flight) run on. Where there is none they skip; Windows CI
//! sets `VICTORY_WARP_REQUIRED=1` (its runners always have WARP), which
//! turns a missing device into a failure, so a test that passed there ran.

use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::{D3D12CreateDevice, ID3D12Device};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
};

/// Whether a test that cannot get its WARP device fails instead of skipping.
pub(crate) fn required() -> bool {
    std::env::var("VICTORY_WARP_REQUIRED").is_ok_and(|v| v == "1")
}

/// A device on WARP, or `None` (the caller skips) where there is none and
/// it is not [`required`].
pub(crate) fn device() -> Option<ID3D12Device> {
    // SAFETY: DXGI and D3D12 creation calls; the only pointer is the out
    // parameter, a local of ours.
    let made = unsafe {
        CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .map_err(|e| format!("no DXGI factory: {e}"))
            .and_then(|factory| {
                factory
                    .EnumWarpAdapter::<IDXGIAdapter1>()
                    .map_err(|e| format!("no WARP adapter: {e}"))
            })
            .and_then(|adapter| {
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)
                    .map_err(|e| format!("no WARP device: {e}"))?;
                device.ok_or_else(|| "no WARP device".to_owned())
            })
    };
    match made {
        Ok(device) => {
            eprintln!("on WARP (software)");
            Some(device)
        }
        Err(e) if required() => panic!("VICTORY_WARP_REQUIRED=1, but {e}"),
        Err(e) => {
            eprintln!("{e}; skipped");
            None
        }
    }
}
