#!/bin/bash
# Starts the Flask backend. On macOS behind a TLS-intercepting corporate proxy,
# outbound HTTPS (Yahoo Finance, Google News, Gemini) fails with self-signed
# certificate errors unless the system keychain CAs are trusted — so we export
# them and merge with certifi's bundle before launching.
set -e
cd "$(dirname "$0")"

PY=./venv/bin/python
[ -x "$PY" ] || PY=python3

if [[ "$OSTYPE" == "darwin"* ]]; then
  BUNDLE="$(pwd)/.ca-bundle.pem"
  security find-certificate -a -p /Library/Keychains/System.keychain > "$BUNDLE" 2>/dev/null || true
  security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain >> "$BUNDLE" 2>/dev/null || true
  CERTIFI=$($PY -c "import certifi; print(certifi.where())" 2>/dev/null || true)
  [ -n "$CERTIFI" ] && cat "$CERTIFI" >> "$BUNDLE"
  export SSL_CERT_FILE="$BUNDLE" CURL_CA_BUNDLE="$BUNDLE" REQUESTS_CA_BUNDLE="$BUNDLE"
fi

exec $PY app.py
