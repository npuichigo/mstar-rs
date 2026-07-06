//! Length-prefixed message framing over a byte stream.
//!
//! Wire format: a 4-byte little-endian `u32` length, then that many bytes of
//! bincode-serialized payload. Simple, self-delimiting, and — unlike mstar's
//! pickle — language-agnostic and safe to parse from untrusted input (a bad
//! length just errors instead of executing code).

use std::io::{Read, Write};

use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;

/// Reject absurd frame lengths early (defends against a desync / bad peer).
const MAX_FRAME_BYTES: u32 = 256 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum FrameError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialize: {0}")]
    Serialize(bincode::Error),
    #[error("frame too large: {0} bytes (max {MAX_FRAME_BYTES})")]
    TooLarge(u32),
}

/// Serialize `msg` and write it as one length-prefixed frame.
pub fn write_frame<M: Serialize, W: Write>(w: &mut W, msg: &M) -> Result<(), FrameError> {
    let bytes = bincode::serialize(msg).map_err(FrameError::Serialize)?;
    let len = bytes.len() as u32;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(&bytes)?;
    w.flush()?;
    Ok(())
}

/// Read one length-prefixed frame and deserialize it. Returns `Ok(None)` on a
/// clean EOF at a frame boundary (peer closed the stream).
pub fn read_frame<M: DeserializeOwned, R: Read>(r: &mut R) -> Result<Option<M>, FrameError> {
    let mut len_buf = [0u8; 4];
    match r.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_le_bytes(len_buf);
    if len > MAX_FRAME_BYTES {
        return Err(FrameError::TooLarge(len));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    let msg = bincode::deserialize(&buf).map_err(FrameError::Serialize)?;
    Ok(Some(msg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    enum Msg {
        Ping(u64),
        Data { name: String, dims: Vec<i64> },
    }

    #[test]
    fn roundtrip_multiple_frames() {
        let mut buf: Vec<u8> = Vec::new();
        let msgs = vec![
            Msg::Ping(42),
            Msg::Data {
                name: "x".into(),
                dims: vec![1, 2, 3],
            },
            Msg::Ping(7),
        ];
        for m in &msgs {
            write_frame(&mut buf, m).unwrap();
        }
        let mut cur = std::io::Cursor::new(buf);
        for expected in &msgs {
            let got: Msg = read_frame(&mut cur).unwrap().unwrap();
            assert_eq!(&got, expected);
        }
        // Clean EOF at a boundary.
        assert!(read_frame::<Msg, _>(&mut cur).unwrap().is_none());
    }

    #[test]
    fn rejects_oversized_length() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(MAX_FRAME_BYTES + 1).to_le_bytes());
        let mut cur = std::io::Cursor::new(buf);
        assert!(matches!(
            read_frame::<Msg, _>(&mut cur),
            Err(FrameError::TooLarge(_))
        ));
    }
}
