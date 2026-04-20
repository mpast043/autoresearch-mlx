"""Security agent for vulnerability scanning of generated solutions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.agents.base import BaseAgent


# OWASP Top 10 patterns (2021) - mapped to code patterns
OWASP_PATTERNS = {
    "A01:2021-Broken_Access_Control": [
        r"\buser_id\b.*=.*request\.params",
        r"if.*user.*is_admin",
        r"\.filter.*user_id.*=",
        r"@app\.route.*admin",
        r"role.*check.*missing",
    ],
    "A02:2021-Cryptographic_Failures": [
        r"md5\(",
        r"sha1\(",
        r"\.encode\('utf-8'\)",
        r"secrets\.token_hex.*length.*<.*32",
        r"hardcoded.*password",
        r"api[_-]?key.*=.*['\"]",
    ],
    "A03:2021-Injection": [
        r"execute\([^f].*\%",
        r"cursor\.execute.*\+",
        r"f\".*SELECT.*{",
        r"\.format\([^)]*\$",
        r"eval\(",
        r"exec\(",
    ],
    "A04:2021-Insecure_Design": [
        r"no.*rate.*limit",
        r"missing.*auth",
        r"csrf.*disabled",
        r"\.set_cookie.*secure.*=.*False",
    ],
    "A05:2021-Security_Misconfiguration": [
        r"debug.*=.*True",
        r"cors.*allow.*\*",
        r"\.env.*in.*git",
        r"default.*password",
    ],
    "A06:2021-Vulnerable_Components": [
        r"pip install.*@latest",
        r"npm install.*@latest",
        r"requirements\.txt.*==",
    ],
    "A07:2021-Auth_Failures": [
        r"password.*==.*None",
        r"if.*not.*password",
        r"session.*not.*verify",
        r"jwt.*verify.*=.*False",
    ],
    "A08:2021-Software_Data_Integrity_Failures": [
        r"pickle\.loads",
        r"yaml\.load.*Loader.*=.*FullLoader",
        r"assert.*not.*reached",
    ],
    "A09:2021-Security_Logging_Failures": [
        r"log.*password",
        r"log.*token",
        r"logging.*debug.*sql",
    ],
    "A10:2021-SSRF": [
        r"requests\.get.*url.*user",
        r"urllib\.open.*url.*user",
        r"subprocess.*url",
    ],
}

# Priority markers like The Agency's code reviewer
PRIORITY_BLOCKER = "🔴"
PRIORITY_SUGGESTION = "🟡"
PRIORITY_NIT = "💭"


@dataclass
class SecurityFinding:
    """A security vulnerability finding."""

    category: str
    severity: str  # blocker, suggestion, nit
    line_number: int | None
    code_snippet: str
    description: str
    cwe_id: str | None = None
    recommendation: str = ""


@dataclass
class SecurityReport:
    """Security scan report for a solution."""

    solution_id: str
    wedge_id: int
    scan_timestamp: float
    findings: list[SecurityFinding] = field(default_factory=list)
    blocker_count: int = 0
    suggestion_count: int = 0
    nit_count: int = 0
    safe: bool = True


class SecurityAgent(BaseAgent):
    """Security-focused agent for vulnerability scanning.

    Scans generated solutions, specs, and code for OWASP Top 10
    vulnerabilities and other security issues.
    """

    def __init__(self, db=None, message_queue=None, config: dict[str, Any] | None = None):
        super().__init__("SecurityAgent")
        self.db = db
        self.config = config or {}
        self._patterns = OWASP_PATTERNS
        self._min_severity = self.config.get("min_severity", "nit")

    async def process(self, message) -> dict[str, Any]:
        """Process a security scan request."""
        payload = message.payload if hasattr(message, "payload") else message
        code = payload.get("code")
        spec = payload.get("spec")
        file_name = payload.get("file_name", "unknown")

        if code:
            report = self.scan_code(code, file_name)
            return {"report": report, "formatted": self.format_report(report)}
        elif spec:
            report = self.scan_solution_spec(spec)
            return {"report": report, "formatted": self.format_report(report)}
        return {"error": "No code or spec provided"}

    def scan_code(self, code: str, file_name: str = "unknown") -> SecurityReport:
        """Scan code for security vulnerabilities.

        Args:
            code: The source code to scan
            file_name: Optional file name for context

        Returns:
            SecurityReport with all findings
        """
        findings = []
        lines = code.split("\n")

        for line_num, line in enumerate(lines, start=1):
            for category, patterns in self._patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        severity = self._determine_severity(category)
                        if self._should_report(severity):
                            finding = SecurityFinding(
                                category=category,
                                severity=severity,
                                line_number=line_num,
                                code_snippet=line.strip()[:100],
                                description=self._get_description(category, pattern),
                                cwe_id=self._get_cwe(category),
                                recommendation=self._get_recommendation(category),
                            )
                            findings.append(finding)

        # Count by severity
        blocker_count = sum(1 for f in findings if f.severity == "blocker")
        suggestion_count = sum(1 for f in findings if f.severity == "suggestion")
        nit_count = sum(1 for f in findings if f.severity == "nit")

        return SecurityReport(
            solution_id=file_name,
            wedge_id=0,
            scan_timestamp=0,
            findings=findings,
            blocker_count=blocker_count,
            suggestion_count=suggestion_count,
            nit_count=nit_count,
            safe=blocker_count == 0,
        )

    def scan_solution_spec(self, spec: dict[str, Any]) -> SecurityReport:
        """Scan a solution specification for security issues.

        Args:
            spec: The solution specification dict

        Returns:
            SecurityReport with findings
        """
        findings = []

        # Check for missing security components
        required_security = ["auth", "encryption", "input_validation", "rate_limiting"]
        for component in required_security:
            if component not in spec and component not in str(spec.get("features", [])):
                findings.append(
                    SecurityFinding(
                        category="A04:2021-Insecure_Design",
                        severity="suggestion",
                        line_number=None,
                        code_snippet=f"Missing: {component}",
                        description=f"Security component '{component}' not found in spec",
                        recommendation=f"Add {component} to solution design",
                    )
                )

        # Check API keys in spec
        spec_str = str(spec)
        if re.search(r"api[_-]?key", spec_str, re.IGNORECASE):
            findings.append(
                SecurityFinding(
                    category="A02:2021-Cryptographic_Failures",
                    severity="blocker",
                    line_number=None,
                    code_snippet="api_key in spec",
                    description="API key detected in specification",
                    recommendation="Use environment variables, not hardcoded values",
                )
            )

        # Check for hardcoded secrets
        secret_patterns = ["password", "secret", "token", "private_key"]
        for secret in secret_patterns:
            if re.search(rf"{secret}\s*=\s*['\"][^$]", spec_str, re.IGNORECASE):
                findings.append(
                    SecurityFinding(
                        category="A02:2021-Cryptographic_Failures",
                        severity="blocker",
                        line_number=None,
                        code_snippet=f"hardcoded {secret}",
                        description=f"Hardcoded {secret} detected in specification",
                        recommendation=f"Use environment variables for {secret}",
                    )
                )

        blocker_count = sum(1 for f in findings if f.severity == "blocker")
        suggestion_count = sum(1 for f in findings if f.severity == "suggestion")

        return SecurityReport(
            solution_id=spec.get("id", "unknown"),
            wedge_id=spec.get("wedge_id", 0),
            scan_timestamp=0,
            findings=findings,
            blocker_count=blocker_count,
            suggestion_count=suggestion_count,
            nit_count=0,
            safe=blocker_count == 0,
        )

    def _determine_severity(self, category: str) -> str:
        """Determine severity based on category."""
        high_risk = ["A01", "A02", "A03", "A07", "A10"]
        medium_risk = ["A04", "A05", "A06", "A08"]
        low_risk = ["A09"]

        for risk in high_risk:
            if risk in category:
                return "blocker"
        for risk in medium_risk:
            if risk in category:
                return "suggestion"
        for risk in low_risk:
            if risk in category:
                return "nit"
        return "suggestion"

    def _should_report(self, severity: str) -> bool:
        """Check if severity meets minimum threshold."""
        severity_order = {"blocker": 3, "suggestion": 2, "nit": 1}
        min_level = severity_order.get(self._min_severity, 1)
        return severity_order.get(severity, 0) >= min_level

    def _get_cwe(self, category: str) -> str | None:
        """Map category to CWE ID."""
        cwe_map = {
            "A01:2021-Broken_Access_Control": "CWE-284",
            "A02:2021-Cryptographic_Failures": "CWE-310",
            "A03:2021-Injection": "CWE-89",
            "A04:2021-Insecure_Design": "CWE-829",
            "A05:2021-Security_Misconfiguration": "CWE-16",
            "A06:2021-Vulnerable_Components": "CWE-1104",
            "A07:2021-Auth_Failures": "CWE-287",
            "A08:2021-Software_Data_Integrity_Failures": "CWE-502",
            "A09:2021-Security_Logging_Failures": "CWE-532",
            "A10:2021-SSRF": "CWE-918",
        }
        return cwe_map.get(category)

    def _get_description(self, category: str, pattern: str) -> str:
        """Get human-readable description for the finding."""
        desc_map = {
            "A01:2021-Broken_Access_Control": "Potential broken access control - verify authorization",
            "A02:2021-Cryptographic_Failures": "Weak cryptographic implementation detected",
            "A03:2021-Injection": "Potential injection vulnerability",
            "A04:2021-Insecure_Design": "Missing security design element",
            "A05:2021-Security_Misconfiguration": "Security misconfiguration detected",
            "A06:2021-Vulnerable_Components": "Potentially vulnerable dependency",
            "A07:2021-Auth_Failures": "Authentication weakness detected",
            "A08:2021-Software_Data_Integrity_Failures": "Data integrity concern",
            "A09:2021-Security_Logging_Failures": "Sensitive data in logs",
            "A10:2021-SSRF": "Potential server-side request forgery",
        }
        return desc_map.get(category, f"Security issue in {category}")

    def _get_recommendation(self, category: str) -> str:
        """Get remediation recommendation."""
        rec_map = {
            "A01:2021-Broken_Access_Control": "Implement proper authorization checks at every layer",
            "A02:2021-Cryptographic_Failures": "Use strong hashing (bcrypt/argon2) and proper key management",
            "A03:2021-Injection": "Use parameterized queries, input validation, and output encoding",
            "A04:2021-Insecure_Design": "Incorporate security requirements in design phase",
            "A05:2021-Security_Misconfiguration": "Review default configurations and harden settings",
            "A06:2021-Vulnerable_Components": "Audit and update dependencies regularly",
            "A07:2021-Auth_Failures": "Implement proper authentication flows with session management",
            "A08:2021-Software_Data_Integrity_Failures": "Use safe deserialization and integrity checks",
            "A09:2021-Security_Logging_Failures": "Redact sensitive data from logs, use structured logging",
            "A10:2021-SSRF": "Validate and sanitize all URL inputs, restrict URL schemes",
        }
        return rec_map.get(category, "Review security requirements")

    def format_report(self, report: SecurityReport) -> str:
        """Format security report with emoji markers."""
        lines = [
            f"# Security Scan Report",
            f"",
            f"Solution: {report.solution_id}",
            f"Wedge ID: {report.wedge_id}",
            f"Findings: {len(report.findings)} total",
            f"  {PRIORITY_BLOCKER} Blockers: {report.blocker_count}",
            f"  {PRIORITY_SUGGESTION} Suggestions: {report.suggestion_count}",
            f"  {PRIORITY_NIT} Nits: {report.nit_count}",
            f"",
            f"Safe to deploy: {'✅ Yes' if report.safe else '❌ No'}",
            f"",
        ]

        if report.findings:
            lines.append("## Findings")
            for f in report.findings:
                marker = {
                    "blocker": PRIORITY_BLOCKER,
                    "suggestion": PRIORITY_SUGGESTION,
                    "nit": PRIORITY_NIT,
                }.get(f.severity, "?")

                lines.append(f"")
                lines.append(f"{marker} **{f.severity.upper()}**: {f.category}")
                if f.line_number:
                    lines.append(f"   Line {f.line_number}: {f.code_snippet}")
                else:
                    lines.append(f"   {f.code_snippet}")
                lines.append(f"   {f.description}")
                if f.recommendation:
                    lines.append(f"   Recommendation: {f.recommendation}")
                if f.cwe_id:
                    lines.append(f"   CWE: {f.cwe_id}")

        return "\n".join(lines)