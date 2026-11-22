//! mdchat core library: Markdown conversation parsing, hierarchical context
//! assembly, and an OpenAI-compatible streaming LLM client.
//!
//! The binary in `main.rs` is a thin CLI over these modules; the web server
//! (a later milestone) will reuse them as well.

pub mod context;
pub mod llm;
pub mod parser;
