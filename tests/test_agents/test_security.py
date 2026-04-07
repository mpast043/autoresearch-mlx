"""Tests for SecurityAgent."""

import pytest

from src.agents.security import (
    SecurityAgent,
    SecurityFinding,
    SecurityReport,
    PRIORITY_BLOCKER,
    PRIORITY_SUGGESTION,
    PRIORITY_NIT,
)


class TestSecurityAgent:
    """Test suite for SecurityAgent."""

    def test_scan_code_injection(self):
        """Test detection of SQL injection vulnerability."""
        agent = SecurityAgent()
        code = """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
"""
        report = agent.scan_code(code, "test.py")

        # Should detect injection
        assert report.blocker_count > 0
        assert not report.safe
        injection_findings = [f for f in report.findings if "Injection" in f.category]
        assert len(injection_findings) > 0

    def test_scan_code_hardcoded_secret(self):
        """Test detection of hardcoded API key."""
        agent = SecurityAgent()
        code = """
API_KEY = "sk-1234567890abcdef"
SECRET = "my_secret_password"
"""
        report = agent.scan_code(code, "config.py")

        # Should detect hardcoded secrets
        assert report.blocker_count > 0
        secrets_found = [f for f in report.findings if "Cryptographic" in f.category]
        assert len(secrets_found) > 0

    def test_scan_code_weak_crypto(self):
        """Test detection of weak cryptographic functions."""
        agent = SecurityAgent()
        code = """
import hashlib
password_hash = hashlib.md5(password.encode()).hexdigest()
"""
        report = agent.scan_code(code, "auth.py")

        # Should detect weak crypto
        md5_findings = [f for f in report.findings if "md5" in f.code_snippet]
        assert len(md5_findings) > 0

    def test_scan_code_safe(self):
        """Test that safe code passes."""
        agent = SecurityAgent()
        code = """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = ?"
    cursor.execute(query, (user_id,))
"""
        report = agent.scan_code(code, "safe.py")

        # Should have no blockers
        assert report.blocker_count == 0
        assert report.safe

    def test_scan_solution_spec_with_api_key(self):
        """Test scanning a spec with hardcoded API key."""
        agent = SecurityAgent()
        spec = {
            "id": "solution-1",
            "wedge_id": 1,
            "title": "Test Solution",
            "api_key": "sk-secret123",
        }
        report = agent.scan_solution_spec(spec)

        assert report.blocker_count > 0
        assert not report.safe

    def test_scan_solution_spec_missing_security(self):
        """Test scanning a spec missing security components."""
        agent = SecurityAgent()
        spec = {
            "id": "solution-1",
            "wedge_id": 1,
            "title": "Test Solution",
            "features": ["user_auth"],
        }
        report = agent.scan_solution_spec(spec)

        # Should flag missing security components
        assert report.suggestion_count > 0

    def test_format_report(self):
        """Test report formatting with emoji markers."""
        agent = SecurityAgent()
        report = SecurityReport(
            solution_id="test.py",
            wedge_id=1,
            scan_timestamp=0,
            findings=[
                SecurityFinding(
                    category="A03:2021-Injection",
                    severity="blocker",
                    line_number=5,
                    code_snippet="query = f'SELECT * FROM users'",
                    description="SQL injection",
                    cwe_id="CWE-89",
                )
            ],
            blocker_count=1,
            suggestion_count=0,
            nit_count=0,
            safe=False,
        )

        formatted = agent.format_report(report)

        assert PRIORITY_BLOCKER in formatted
        assert "test.py" in formatted
        assert "CWE-89" in formatted

    def test_min_severity_filter(self):
        """Test severity filtering."""
        agent = SecurityAgent(config={"min_severity": "blocker"})
        code = """
# This has a suggestion but not a blocker
def unused():
    pass

# This has a blocker
API_KEY = "secret"
"""
        report = agent.scan_code(code)

        # Should only show blockers due to min_severity filter
        assert all(f.severity == "blocker" for f in report.findings)


class TestSecurityFinding:
    """Test SecurityFinding dataclass."""

    def test_finding_creation(self):
        """Test creating a security finding."""
        finding = SecurityFinding(
            category="A03:2021-Injection",
            severity="blocker",
            line_number=42,
            code_snippet="query = f'SELECT * FROM users'",
            description="SQL injection vulnerability",
            cwe_id="CWE-89",
            recommendation="Use parameterized queries",
        )

        assert finding.category == "A03:2021-Injection"
        assert finding.severity == "blocker"
        assert finding.line_number == 42
        assert finding.cwe_id == "CWE-89"