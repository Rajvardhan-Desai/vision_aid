import smtplib
import logging
from email.mime.text import MIMEText
from typing import Optional, Tuple
from . import settings

log = logging.getLogger("vision_aid")

def get_gps_location() -> Optional[Tuple[float,float]]:
    return None  # placeholder

def send_email(subject: str, body: str,
               sender: Optional[str] = None,
               password: Optional[str] = None,
               dest: Optional[str] = None,
               smtp_server: Optional[str] = None,
               smtp_port: Optional[int] = None) -> bool:
    sender = sender or settings.sender_email()
    password = password or settings.sender_password()
    dest = dest or settings.emergency_contact()
    smtp_server = smtp_server or settings.smtp_server()
    smtp_port = int(smtp_port or settings.smtp_port())

    if not (sender and password and dest):
        log.error("Email not sent: missing .env keys: SENDER_EMAIL/SENDER_PASSWORD/EMERGENCY_CONTACT")
        return False

    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = dest

        with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, [dest], msg.as_string())
        return True
    except Exception as e:
        log.error("Email send failed: %s", e)
        return False
