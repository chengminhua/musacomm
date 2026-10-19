"""Synthetic, CPU-only publication checks; never use real private identifiers."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_public_tree.py"
SPEC = importlib.util.spec_from_file_location("musacomm_publication_check", SCRIPT)
check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check
SPEC.loader.exec_module(check)


class PublicTreeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "src").mkdir()
        (self.root / "docs").mkdir()
        (self.root / "README.md").write_text("# Example\n", encoding="utf-8")

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def codes(self):
        return {finding.code for finding in check.check_tree(self.root)}

    def test_clean_tree_and_loopback_are_allowed(self):
        self.write("README.md", "[usage](docs/usage.md#example)\n")
        self.write("docs/usage.md", "# Example\n[home](../README.md)\n")
        self.write("src/example.py", "address = '127.0.0.1'\n")
        self.assertEqual(check.check_tree(self.root), [])

    @unittest.skipUnless(shutil.which("git"), "Git is needed to verify publication ignore rules")
    def test_git_ignores_generated_roots_but_keeps_repository_skills(self):
        repository = Path(__file__).resolve().parents[1]
        shutil.copyfile(repository / ".gitignore", self.root / ".gitignore")
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True,
                       capture_output=True, text=True)
        skills = ["skills/musacomm-usage/SKILL.md", "skills/musacomm-development/SKILL.md"]
        for path in skills:
            self.write(path, "# Synthetic skill\n")
        command = ["git", "-c", "core.excludesFile=/dev/null", "-C", str(self.root),
                   "check-ignore", "--no-index"]
        included = subprocess.run([*command, *skills], capture_output=True, text=True)
        self.assertEqual(included.returncode, 1, included.stdout + included.stderr)
        generated = ["plan-check/file.txt", "result-check/file.txt", "musacomm-check/file.txt"]
        for path in generated:
            self.write(path, "generated\n")
        ignored = subprocess.run([*command, *generated], capture_output=True, text=True)
        self.assertEqual(ignored.returncode, 0, ignored.stderr)
        self.assertEqual(set(ignored.stdout.splitlines()), set(generated))

    def test_private_networks_and_invalid_addresses(self):
        addresses = [".".join(map(str, parts)) for parts in (
            (10, 3, 7, 8), (172, 20, 4, 9), (192, 168, 6, 7),
        )]
        findings = check.scan_text("\n".join(addresses), "synthetic.md")
        self.assertEqual([item.code for item in findings], ["private-address"] * 3)
        self.assertEqual(check.scan_text("127.0.0.1 999.1.2.3 192.0.2.1", "synthetic.md"), [])

    def test_private_identifiers(self):
        contents = "/" + "home/" + "synthetic-user/" + "work\n" + "worker" + str(12345)
        self.write("docs/example.md", contents)
        self.assertEqual(self.codes(), {"user-home", "private-host"})

    def test_private_key_and_token_detection_do_not_echo_values(self):
        key = "-----BEGIN " + "PRIVATE KEY-----"
        token = "gh" + "p_" + "a" * 40
        findings = check.scan_text(key + "\n" + token, "synthetic.md")
        self.assertEqual({item.code for item in findings}, {"private-key", "credential-token"})
        self.assertNotIn(token, repr(findings))
        self.assertNotIn(key, repr(findings))

    def test_credential_assignments_and_placeholders(self):
        assignment = "api_" + "key = '" + "synthetic-value" + "'"
        self.assertEqual(check.scan_text(assignment, "synthetic.py")[0].code, "credential-assignment")
        placeholder = "api_" + "key = 'YOUR_VALUE_HERE'"
        self.assertEqual(check.scan_text(placeholder, "synthetic.py"), [])

    def test_relative_links_and_code_examples(self):
        self.write("docs/with space.md", "# Example\n")
        self.write("README.md", "\n".join((
            "[good](docs/with%20space.md)", "[remote](https://example.com/page)",
            "[reference]: docs/with%20space.md", "[section](#example)",
            "`[not a link](missing.md)`", "```markdown", "[example](missing.md)", "```",
        )))
        self.assertEqual(self.codes(), set())

    def test_broken_absolute_and_escaping_links(self):
        self.write("README.md", "[missing](docs/missing.md)\n[out](../outside.md)\n[local](/tmp/test.md)\n")
        self.assertEqual(self.codes(), {"broken-link", "escaping-link", "absolute-local-link"})

    def test_skill_relative_documentation_links(self):
        self.write("docs/usage.md", "# Usage\n")
        for name in ("musacomm-usage", "musacomm-development"):
            self.write(f"skills/{name}/SKILL.md", "\n".join((
                "# Skill", "[project](../../README.md)",
                "[usage](../../docs/usage.md)", "[interface](agents/openai.yaml)",
            )))
            self.write(f"skills/{name}/agents/openai.yaml", 'interface:\n  display_name: "Example"\n')
        self.assertEqual(check.check_tree(self.root), [])

    def test_agents_documentation_links_are_checked(self):
        self.write("skills/musacomm-usage/SKILL.md", "# Usage\n")
        self.write("AGENTS.md", "[skill](skills/musacomm-usage/SKILL.md)\n")
        self.assertEqual(check.check_tree(self.root), [])
        self.write("AGENTS.md", "[missing](skills/musacomm-development/SKILL.md)\n")
        findings = check.check_tree(self.root)
        self.assertEqual([(item.path, item.code) for item in findings],
                         [("AGENTS.md", "broken-link")])

    def test_nested_skill_artifact_prefixes_cannot_bypass_scan(self):
        private_identifier = "worker" + str(12345)
        expected = set()
        for directory in ("plan-draft", "result-run", "results", "musacomm-hidden"):
            prefix = f"skills/musacomm-usage/references/{directory}"
            self.write(prefix + "/private.md", private_identifier)
            self.write(prefix + "/links.md", "[missing](missing.md)\n")
            self.write(prefix + "/artifact.bin", "synthetic\n")
            expected.update((
                (prefix + "/private.md", "private-host"),
                (prefix + "/links.md", "broken-link"),
                (prefix + "/artifact.bin", "vendor-or-binary-file"),
            ))
        findings = check.check_tree(self.root)
        self.assertEqual({(item.path, item.code) for item in findings}, expected)

    def test_forbidden_headers_binaries_logs_and_archives(self):
        for name in ("mccl.h", "libexample.so.1", "trace.log", "results.tar.gz"):
            self.write("src/" + name, "synthetic\n")
        self.assertEqual(self.codes(), {"vendor-or-binary-file", "raw-result-or-archive"})

    def test_generated_directories_are_outside_scan(self):
        key = "-----BEGIN " + "PRIVATE KEY-----"
        for directory in (".git", ".venv", "build", "dist", "results", "plan-ci",
                          "result-run", "musacomm-run", "src/example.egg-info", "tests/__pycache__",
                          "skills/musacomm-usage/__pycache__",
                          "skills/musacomm-development/example.egg-info"):
            self.write(directory + "/ignored.txt", key)
        self.assertEqual(self.codes(), set())

    def test_symlink_is_rejected_without_reading_target(self):
        path = self.write("docs/real.md", "# Example\n")
        (self.root / "docs" / "linked.md").symlink_to(path)
        self.assertEqual(self.codes(), {"symlink"})

    def test_unexpected_top_level_and_binary_content(self):
        self.write("unapproved.txt", "synthetic\n")
        self.write("src/data.py", "\x00")
        self.assertEqual(self.codes(), {"unexpected-root-path", "binary-content"})

    def test_invalid_utf8_is_rejected(self):
        (self.root / "src" / "invalid.py").write_bytes(bytes([255]))
        self.assertEqual(self.codes(), {"unreadable-text"})


if __name__ == "__main__":
    unittest.main()
