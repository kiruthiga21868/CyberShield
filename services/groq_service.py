"""Cybersecurity education chatbot: Groq LLM with a local offline fallback.

Design rules
------------
* Groq is optional. Without ``GROQ_API_KEY`` the service answers from a local
  keyword-matched knowledge base, so the feature is never dead in the UI.
* The API key is read from the environment, never logged, never returned.
* Network errors, rate limits and authentication failures fall back to the
  local knowledge base instead of surfacing an error to the user.
* The system prompt restricts the assistant to defensive cybersecurity
  education and forbids guarantees or professional incident-response advice.
* Conversation history lives in the caller's Streamlit session only; this
  module never persists anything.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from config import (
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
    GROQ_TIMEOUT,
    get_groq_api_key,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are CyberShield AI, a cybersecurity education assistant.

Your purpose is to help everyday users understand cyber threats and protect
themselves online. You explain concepts in plain language and give practical,
actionable advice.

Topics you cover:
- phishing, social engineering, smishing, vishing and spear phishing
- spam, scam messages and how to report them
- password and account security
- QR code phishing ("quishing")
- suspicious links, domains and URLs
- malware, ransomware and browser safety
- two-factor authentication and safe browsing habits
- what to do after clicking a suspicious link or entering credentials

Hard limits you must respect:
- Never claim that a URL, file or message is definitively safe or malicious.
  Say what the indicators suggest and recommend independent verification.
- Never provide professional incident response, digital forensics or legal
  advice. Explain general principles, then suggest contacting the relevant
  organisation, a security professional or the local cybercrime authority.
- Never help with attacking systems, bypassing authentication, exploiting
  software, malware development, credential theft or evading detection.
  If asked, briefly decline and redirect to defensive guidance.
- Never ask for, request or repeat passwords, full card numbers, one-time
  codes, API keys or other secrets. If a user pastes one, tell them to change
  it and never repeat it back.
- Do not invent statistics, incident details or product capabilities.

Style: concise, calm and practical. Prefer short paragraphs and bullet points.
Aim for under 200 words unless the user asks for depth.
"""


# --------------------------------------------------------------------------
# Offline knowledge base
# --------------------------------------------------------------------------
_KB: tuple[tuple[tuple[str, ...], str], ...] = (
    (("phishing",),
     "Phishing is a social-engineering attack in which an attacker impersonates "
     "a trusted brand, person or service to trick you into revealing "
     "information or clicking something harmful.\n\n"
     "**Warning signs**\n"
     "- Urgency or threats: 'your account will be closed today'\n"
     "- A sender address that does not match the organisation it claims to be\n"
     "- Spelling mistakes and odd formatting\n"
     "- Links whose real destination differs from the visible text\n"
     "- Unexpected attachments or requests for credentials, card details or "
     "one-time codes\n\n"
     "**What to do**\n"
     "- Do not click the link. Navigate to the site yourself by typing the "
     "address or using a saved bookmark.\n"
     "- Verify through a channel you already trust.\n"
     "- Report the message, then delete it."),
    (("quishing", "qr code", "qrcode", "qr"),
     "QR phishing ('quishing') is a phishing attack that hides a malicious link "
     "inside a QR code image.\n\n"
     "**Why it works**\n"
     "A QR code hides its destination from you and from most link scanners, and "
     "physical posters, emails and menus are less likely to be inspected.\n\n"
     "**What to do**\n"
     "- Scan with CyberShield AI's QR Scanner before visiting the destination.\n"
     "- Check the domain carefully once the page opens.\n"
     "- Be suspicious of QR codes on unsolicited emails, parking meters and "
     "posters offering gifts.\n"
     "- On desktop, preview the URL before loading it."),
    (("spam",),
     "Spam is unsolicited bulk email or SMS. It is usually annoying rather than "
     "dangerous, but it is a common delivery route for phishing and malware.\n\n"
     "**How to spot it**\n"
     "- Generic greetings, all-caps text and excessive exclamation marks\n"
     "- Prize, lottery, urgency or 'your parcel is waiting' language\n"
     "- Many links, a mismatched sender domain, or a shortened link\n\n"
     "**What to do**\n"
     "- Report it as spam in your mail client and delete it.\n"
     "- Never reply, even to unsubscribe, because that confirms your address is "
     "active."),
    (("password",),
     "A strong password is long, unique per site and generated randomly.\n\n"
     "**Guidance**\n"
     "- Aim for 14-16 characters or more; length beats symbol count.\n"
     "- Avoid dictionary words, names, dates and obvious substitutions like "
     "'P@ssw0rd'.\n"
     "- Never reuse a password. Reuse is what turns one breach into many.\n"
     "- Use a reputable password manager so a unique password per site is "
     "practical.\n"
     "- Use CyberShield AI's Password Analyzer to check a candidate before you "
     "save it.\n\n"
     "CyberShield AI analyses passwords entirely on your own device and never "
     "sends or stores them."),
    (("clicked", "clicked a link", "already clicked", "followed"),
     "You clicked a suspicious link. Acting quickly matters more than panicking.\n\n"
     "**Do this now**\n"
     "1. Close the page. Do not interact further, and do not download anything.\n"
     "2. If you entered a password, change it from the real site (type the "
     "address yourself) and change it everywhere you reused it.\n"
     "3. If you entered card details, contact your bank's fraud line.\n"
     "4. Enable two-factor authentication on the affected account.\n"
     "5. Run a reputable antivirus or anti-malware scan on your device.\n"
     "6. Check your bank and email account for unfamiliar activity.\n"
     "7. Report the message to your IT department or email provider.\n\n"
     "Simply visiting a page is often low risk; entering credentials or "
     "downloading a file is what raises the danger."),
    (("malware", "virus", "ransomware", "trojan"),
     "Malware is software written to damage a system, steal data or hold files "
     "for ransom.\n\n"
     "**How it arrives**\n"
     "- Infected attachments or fake invoice documents\n"
     "- Cracked software and 'free' downloads\n"
     "- Malicious adverts that redirect to exploit kits\n"
     "- QR codes and shortened links\n\n"
     "**Prevention**\n"
     "- Keep the operating system and browser updated.\n"
     "- Use reputable antivirus and keep it enabled.\n"
     "- Never run a file you were not expecting, especially .exe, .bat or .scr.\n"
     "- Back up important files to offline or versioned storage."),
    (("two-factor", "2fa", "mfa", "multi-factor", "otp"),
     "Two-factor authentication (2FA) adds a second proof of identity, usually a "
     "code from an app or a hardware key.\n\n"
     "**Guidance**\n"
     "- Prefer an authenticator app or a hardware key over SMS, which is "
     "vulnerable to SIM-swapping.\n"
     "- Never share a one-time code with anyone, including someone claiming to be "
     "from support.\n"
     "- Enable 2FA on your email provider first; email is the master key that "
     "resets every other account.\n"
     "- Store backup codes somewhere safe and offline."),
    (("suspicious link", "weird link", "short link", "tiny url"),
     "When a link looks wrong, check it before clicking.\n\n"
     "**What to look at**\n"
     "- The real domain, not the visible link text\n"
     "- Misspellings of popular brands (paypa1.com, micros0ft.co)\n"
     "- Abused free top-level domains such as .tk, .ml, .ga, .cf and .zip\n"
     "- Raw IP addresses instead of domain names\n"
     "- Excessive subdomains or a very long address\n\n"
     "**What to do**\n"
     "Paste the URL into CyberShield AI's URL Scanner. It combines a Random "
     "Forest classifier with transparent structural checks, and can be enriched "
     "with VirusTotal reputation data."),
    (("public wifi", "public wi-fi", "coffee shop"),
     "Public Wi-Fi is usually unencrypted, so other people on the same network "
     "may be able to see traffic.\n\n"
     "**Safer habits**\n"
     "- Avoid banking or other sensitive work on public networks.\n"
     "- Use a trusted VPN to encrypt your traffic.\n"
     "- Prefer HTTPS (look for the padlock) and never ignore certificate "
     "warnings.\n"
     "- Turn off automatic Wi-Fi joining and file sharing.\n"
     "- On phones, prefer mobile data over unknown hotspots for sensitive tasks."),
    (("social engineering",),
     "Social engineering is the human side of hacking: manipulating people rather "
     "than breaking software.\n\n"
     "**Common techniques**\n"
     "- Pretexting: inventing a believable scenario, such as IT support\n"
     "- Urgency and fear: an account or payment is about to be lost\n"
     "- Authority: pretending to be a bank, employer or government\n"
     "- Reciprocity: offering a small free gift to build obligation\n"
     "- Misdirection: burying the request in a long, cluttered message\n\n"
     "**Defence**\n"
     "Slow down. Verify identity through a separate channel you initiate "
     "yourself, never one supplied by the message."),
    (("vpn",),
     "A VPN encrypts your traffic between your device and the VPN server, which "
     "helps on untrusted networks. It does not make you anonymous, and a bad VPN "
     "can see your traffic.\n\n"
     "**Points to check**\n"
     "- A no-logs policy that has been independently audited\n"
     "- A clear kill switch and DNS-leak protection\n"
     "- Reputable ownership and a published transparency report\n\n"
     "It is not a substitute for HTTPS, updates or sensible account hygiene."),
    (("browser",),
     "Browser safety basics:\n\n"
     "- Keep the browser and extensions updated; most exploited flaws are patched.\n"
     "- Treat 'update your browser now!' interstitials as likely fake; type the "
     "address yourself instead.\n"
     "- Never allow unexpected notification or location permissions.\n"
     "- Clear cookies and site data for sites you no longer use.\n"
     "- Watch the padlock, but remember it proves encryption, not honesty: an "
     "https:// phishing site still gets a padlock."),
    (("report", "reporting"),
     "If you have been targeted, report it so others are protected.\n\n"
     "- Email: mark as phishing in your mail client, and report phishing in "
     "Gmail, Outlook or your provider's settings.\n"
     "- SMS: forward spam texts to your carrier's spam-reporting number.\n"
     "- Money lost or credentials stolen: contact your bank's fraud line "
     "immediately.\n"
     "- Serious or business incidents: notify your IT team or local cybercrime "
     "authority.\n\n"
     "Reporting is what turns an individual attack into a pattern others can see."),
    (("phishing", "safe", "is it safe", "check a link", "scan a link"),
     "You can analyse a link with CyberShield AI's URL Scanner. It combines a "
     "Random Forest classifier trained on URL features with transparent structural "
     "checks, and can be enriched with VirusTotal reputation data.\n\n"
     "Remember that no automated tool can give a guarantee. Treat a result as one "
     "signal, and verify important destinations through a channel you trust."),
    (("certificate", "https", "padlock"),
     "HTTPS proves that traffic to a site is encrypted and that the certificate "
     "matches the domain. It does **not** prove the site is honest or safe.\n\n"
     "Phishing sites routinely obtain free HTTPS certificates, so a padlock is "
     "necessary but not sufficient. Check the domain name itself, and be wary of "
     "look-alike spellings."),
)


@dataclass
class ChatReply:
    """A single chatbot answer plus the mode that produced it."""

    text: str
    mode: str  # "ai" | "offline"
    error: str | None = None

    @property
    def is_ai(self) -> bool:
        return self.mode == "ai"

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "mode": self.mode, "error": self.error}


def offline_answer(question: str) -> str:
    """Answer from the local knowledge base using keyword matching."""
    text = (question or "").lower().strip()
    if not text:
        return (
            "Ask me about phishing, spam, password security, QR codes, suspicious "
            "links, malware, two-factor authentication, or what to do after "
            "clicking a suspicious link."
        )

    best: tuple[int, str] | None = None
    for keywords, answer in _KB:
        # Longer keywords are more specific, so "report" should beat the
        # incidental "spam" match in "how do I report spam".
        score = sum(len(keyword) for keyword in keywords if keyword in text)
        if score and (best is None or score > best[0]):
            best = (score, answer)

    if best is not None:
        return best[1]

    # No keyword matched: answer the generic question instead of shrugging.
    if re.search(r"\b(url|link|website|domain)\b", text):
        return (
            "For a specific link, use CyberShield AI's URL Scanner: paste the "
            "address and read the risk level together with the detected "
            "indicators.\n\n"
            "In general, check the real domain (not the visible link text), look "
            "for misspellings of popular brands, raw IP addresses, abused free "
            "top-level domains such as .tk or .zip, and unusually long addresses."
        )

    return (
        "I do not have a specific answer for that in my offline knowledge base, "
        "and the AI assistant is not configured right now.\n\n"
        "I can help with:\n"
        "- phishing, smishing, vishing and social engineering\n"
        "- spam and scam messages\n"
        "- password strength and account security\n"
        "- QR code phishing ('quishing')\n"
        "- suspicious links and safe browsing\n"
        "- what to do after clicking a suspicious link\n"
        "- malware, ransomware and two-factor authentication\n\n"
        "For a live security incident, contact your IT department, your bank's "
        "fraud line, or your local cybercrime authority."
    )


class GroqService:
    """Thin wrapper over the Groq chat-completions API with a local fallback."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client: Any | None = None

    @property
    def api_key(self) -> str | None:
        return self._api_key if self._api_key is not None else get_groq_api_key()

    def is_available(self) -> bool:
        """True when a key is configured. No network call is made."""
        return bool(self.api_key)

    @property
    def mode_label(self) -> str:
        return "AI Chatbot (Groq)" if self.is_available() else "Offline Cybersecurity Assistant"

    def _get_client(self) -> Any | None:
        """Build the Groq client lazily; returns None if the SDK is absent."""
        if self._client is not None:
            return self._client
        try:
            from groq import Groq
        except ImportError:
            logger.warning("The 'groq' package is not installed.")
            return None
        try:
            self._client = Groq(api_key=self.api_key)
        except Exception as exc:  # noqa: BLE001 - constructor validation
            logger.warning("Groq client could not be created: %s",
                           type(exc).__name__)
            return None
        return self._client

    def ask(self, question: str, history: list[dict[str, str]] | None = None) -> ChatReply:
        """Answer ``question``, falling back to the offline knowledge base."""
        question = (question or "").strip()
        if not question:
            return ChatReply(
                text="Please type a cybersecurity question.",
                mode="offline" if not self.is_available() else "ai",
            )

        if not self.is_available():
            return ChatReply(text=offline_answer(question), mode="offline")

        client = self._get_client()
        if client is None:
            return ChatReply(
                text=offline_answer(question),
                mode="offline",
                error="The Groq SDK is not installed; using the offline assistant.",
            )

        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for entry in (history or [])[-8:]:
            role = entry.get("role")
            content = entry.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": str(content)})
        messages.append({"role": "user", "content": question})

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=GROQ_TEMPERATURE,
                max_tokens=GROQ_MAX_TOKENS,
                timeout=GROQ_TIMEOUT,
            )
            text = response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001 - any SDK/network failure
            name = type(exc).__name__
            logger.warning("Groq request failed (%s); using offline answer.", name)
            return ChatReply(
                text=offline_answer(question),
                mode="offline",
                error=f"The AI assistant was unreachable ({name}); "
                      f"answered from the offline knowledge base.",
            )

        if not text or not str(text).strip():
            return ChatReply(
                text=offline_answer(question),
                mode="offline",
                error="The AI assistant returned an empty reply.",
            )

        return ChatReply(text=str(text).strip(), mode="ai")

    def info(self) -> dict[str, Any]:
        """Status block for the UI, containing no secret material."""
        return {
            "available": self.is_available(),
            "mode_label": self.mode_label,
            "model": GROQ_MODEL if self.is_available() else None,
            "offline_topics": len(_KB),
        }


__all__ = ["GroqService", "ChatReply", "offline_answer", "SYSTEM_PROMPT"]