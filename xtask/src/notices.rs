//! The notices a binary of VIctory carries (docs/diagnostic-build.md): the
//! crates compiled into `victory.exe` for a target, each one's license as its
//! manifest states it, and the license files its published package holds
//! (MIT, BSD and Apache-2.0 ask for theirs to go with binaries).
//!
//! The crates are what `cargo tree` resolves from Cargo.lock; their packages
//! are read where Cargo unpacked them (`$CARGO_HOME/registry/src`), so run
//! this after the build that fetched them.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, ensure};

/// A crate the executable is built with.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Crate {
    name: String,
    version: String,
    /// Where it comes from when not crates.io: a path for the workspace's
    /// own crates.
    source: Option<String>,
    license: String,
    repository: String,
}

/// One line of `cargo tree --prefix none --format "{p}|{l}|{r}"`: `name
/// vX.Y.Z [(source)] [(proc-macro)]|license|repository [(*)]`.
fn parse(line: &str) -> Option<Crate> {
    let line = line.trim_end();
    let line = line.strip_suffix(" (*)").unwrap_or(line);
    let mut fields = line.splitn(3, '|');
    let package = fields.next()?.replace(" (proc-macro)", "");
    let license = fields.next()?.trim().to_owned();
    let repository = fields.next().unwrap_or("").trim().to_owned();
    let (name, rest) = package.split_once(' ')?;
    let (version, source) = match rest.split_once(' ') {
        Some((version, source)) => {
            let source = source.trim().trim_start_matches('(').trim_end_matches(')');
            (version, Some(source.to_owned()))
        }
        None => (rest, None),
    };
    Some(Crate {
        name: name.to_owned(),
        version: version.strip_prefix('v')?.to_owned(),
        source,
        license,
        repository,
    })
}

/// Whether a package-root entry is a license, notice or copyright file (or
/// a folder of them).
fn is_license_file(name: &str) -> bool {
    let name = name.to_ascii_uppercase();
    [
        "LICENSE",
        "LICENCE",
        "COPYING",
        "COPYRIGHT",
        "NOTICE",
        "UNLICENSE",
    ]
    .iter()
    .any(|p| name.starts_with(p))
}

/// The license files of a package unpacked at `dir`: (file name, text).
fn license_files(dir: &Path) -> Vec<(String, String)> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !is_license_file(&name) {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            for (inner, text) in license_files_in(&path) {
                found.push((format!("{name}/{inner}"), text));
            }
        } else if let Ok(text) = std::fs::read_to_string(&path) {
            found.push((name, text));
        }
    }
    found
}

/// Every text file in a license folder (`LICENSES/`, as REUSE lays out).
fn license_files_in(dir: &Path) -> Vec<(String, String)> {
    let mut found: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let text = std::fs::read_to_string(e.path()).ok()?;
            Some((e.file_name().to_string_lossy().into_owned(), text))
        })
        .collect();
    found.sort();
    found
}

/// Where Cargo unpacks crates.io packages: one folder per registry index.
fn registry_folders() -> Result<Vec<PathBuf>> {
    let home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            let home = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
            std::env::var_os(home).map(|h| PathBuf::from(h).join(".cargo"))
        })
        .context("no CARGO_HOME and no home folder")?;
    let src = home.join("registry").join("src");
    let folders = std::fs::read_dir(&src)
        .with_context(|| format!("reading {}", src.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    Ok(folders)
}

/// The notices for `crates`: `ours` are the workspace's own (covered by
/// LICENSE), `files` gives a crate's license files.
fn notices(
    target: &str,
    crates: &BTreeSet<Crate>,
    ours: impl Fn(&Crate) -> bool,
    files: impl Fn(&Crate) -> Vec<(String, String)>,
) -> String {
    let mut text = format!(
        "Third-party notices: victory.exe ({target})\n\n\
         victory.exe is VIctory, licensed GPL-3.0-or-later (see LICENSE). It is\n\
         built from the crates below, at the versions Cargo.lock pins, and the\n\
         Rust standard library (MIT OR Apache-2.0, https://github.com/rust-lang/rust).\n\
         Each crate's license is as its manifest states it. The license files its\n\
         published package holds follow the list, each distinct text once, with\n\
         the crates it came with.\n\nCrates\n\n"
    );
    // Each distinct text, with the crates it came with.
    let mut texts: BTreeMap<String, (BTreeSet<String>, BTreeSet<String>)> = BTreeMap::new();
    let mut without = Vec::new();
    for c in crates {
        let id = format!("{} {}", c.name, c.version);
        let mine = ours(c);
        let from = if mine {
            "this project".to_owned()
        } else {
            c.repository.clone()
        };
        text.push_str(&format!("  {id:<34} {:<40} {from}\n", c.license));
        if mine {
            continue;
        }
        let found = files(c);
        if found.is_empty() {
            without.push(id.clone());
        }
        for (name, body) in found {
            let body = body.replace("\r\n", "\n").trim_end().to_owned();
            let (names, ids) = texts.entry(body).or_default();
            names.insert(name);
            ids.insert(id.clone());
        }
    }
    if !without.is_empty() {
        text.push_str(&format!(
            "\nPublished without a license file (their licenses are as listed above;\n\
             the texts are in their repositories): {}\n",
            without.join(", ")
        ));
    }
    text.push_str("\nLicense texts\n");
    for (body, (names, ids)) in texts {
        let names: Vec<_> = names.into_iter().collect();
        let ids: Vec<_> = ids.into_iter().collect();
        text.push_str(&format!(
            "\n==== {} (from {})\n\n{body}\n",
            names.join(", "),
            ids.join(", ")
        ));
    }
    text
}

/// Writes the notices for `victory.exe` built for `target` to `out`.
pub fn run(root: &Path, target: &str, out: &Path) -> Result<()> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let tree = Command::new(cargo)
        .args(["tree", "--locked", "-p", "victory", "--target", target])
        .args([
            "-e",
            "normal",
            "--prefix",
            "none",
            "--format",
            "{p}|{l}|{r}",
        ])
        .current_dir(root)
        .output()
        .context("running cargo tree")?;
    ensure!(
        tree.status.success(),
        "cargo tree failed: {}",
        String::from_utf8_lossy(&tree.stderr)
    );
    let crates: BTreeSet<Crate> = String::from_utf8(tree.stdout)
        .context("cargo tree output is not UTF-8")?
        .lines()
        .filter_map(parse)
        .collect();
    ensure!(
        crates.iter().any(|c| c.name == "victory"),
        "cargo tree did not list victory"
    );
    let registries = registry_folders()?;
    let ours = |c: &Crate| {
        c.source
            .as_deref()
            .is_some_and(|s| Path::new(s).starts_with(root))
    };
    let files = |c: &Crate| {
        let folder = format!("{}-{}", c.name, c.version);
        registries
            .iter()
            .map(|r| r.join(&folder))
            .find(|d| d.is_dir())
            .map(|d| license_files(&d))
            .unwrap_or_default()
    };
    let missing: Vec<_> = crates
        .iter()
        .filter(|c| !ours(c) && c.source.is_some())
        .map(|c| {
            format!(
                "{} {} ({})",
                c.name,
                c.version,
                c.source.as_deref().unwrap_or("")
            )
        })
        .collect();
    ensure!(
        missing.is_empty(),
        "crates from outside crates.io and the workspace, whose packages this does not read: {}",
        missing.join(", ")
    );
    let text = notices(target, &crates, ours, files);
    std::fs::write(out, &text).with_context(|| format!("writing {}", out.display()))?;
    println!(
        "{}: {} crates, {} bytes",
        out.display(),
        crates.len(),
        text.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_lines_parse_into_crates() {
        let registry = parse("anyhow v1.0.104|MIT OR Apache-2.0|https://github.com/dtolnay/anyhow");
        let registry = registry.unwrap();
        assert_eq!(
            (registry.name.as_str(), registry.version.as_str()),
            ("anyhow", "1.0.104")
        );
        assert_eq!(registry.source, None);
        assert_eq!(registry.license, "MIT OR Apache-2.0");
        let again = parse(
            "windows-sys v0.61.2|MIT OR Apache-2.0|https://github.com/microsoft/windows-rs (*)",
        );
        assert_eq!(
            again.unwrap().repository,
            "https://github.com/microsoft/windows-rs"
        );
        let derive = parse(
            "serde_derive v1.0.229 (proc-macro)|MIT OR Apache-2.0|https://github.com/serde-rs/serde",
        );
        let derive = derive.unwrap();
        assert_eq!((derive.version.as_str(), derive.source), ("1.0.229", None));
        let ours = parse(r"victory v0.1.0 (D:\a\VIctory\apps\victory)|GPL-3.0-or-later|").unwrap();
        assert_eq!(ours.source.as_deref(), Some(r"D:\a\VIctory\apps\victory"));
        assert_eq!(ours.repository, "");
        assert_eq!(parse(""), None);
    }

    #[test]
    fn each_license_text_is_given_once_with_its_crates() {
        let krate = |name: &str, source: Option<&str>| Crate {
            name: name.into(),
            version: "1.0.0".into(),
            source: source.map(Into::into),
            license: "MIT".into(),
            repository: format!("https://example.invalid/{name}"),
        };
        let crates: BTreeSet<_> = [
            krate("a", None),
            krate("b", None),
            krate("c", None),
            krate("victory-core", Some("/repo/crates/victory-core")),
        ]
        .into();
        let text = notices(
            "x86_64-pc-windows-msvc",
            &crates,
            |c| c.source.is_some(),
            |c| match c.name.as_str() {
                "a" => vec![("LICENSE-MIT".into(), "MIT text\r\n".into())],
                "b" => vec![("license-mit".into(), "MIT text\n".into())],
                _ => Vec::new(),
            },
        );
        assert_eq!(text.matches("MIT text").count(), 1, "{text}");
        assert!(text.contains("==== LICENSE-MIT, license-mit (from a 1.0.0, b 1.0.0)"));
        assert!(text.contains("victory-core 1.0.0"));
        assert!(text.contains("this project"));
        assert!(text.contains("without a license file"));
        assert!(text.contains("): c 1.0.0\n"), "{text}");
    }

    #[test]
    fn license_files_are_recognised_by_name() {
        for name in [
            "LICENSE",
            "LICENSE-MIT",
            "license-apache-2.0",
            "COPYING",
            "UNLICENSE",
        ] {
            assert!(is_license_file(name), "{name}");
        }
        for name in ["LICENSE_A2", "NOTICE.md", "COPYRIGHT"] {
            assert!(is_license_file(name), "{name}");
        }
        for name in ["README.md", "Cargo.toml", "src", "licensing.rs"] {
            assert!(!is_license_file(name), "{name}");
        }
    }
}
