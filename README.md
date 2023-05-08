# Spamino
Spamino is a spam filter service that runs as middleware between external mail servers (ex. Google, Microsoft, etc.) and your mail server. The application uses 5 different layers of spam filtering to ensure that spam is caught before it reaches your mail server. The following layers are used:

- TXT Record Check
- Email IP/Domain Check (SBL and DBL)
- Ensures any URLs in the email are not malicious
- Custom built ML model which checks the context of emails (title and body)
- Custom built ML model which checks the content inside all links in the email