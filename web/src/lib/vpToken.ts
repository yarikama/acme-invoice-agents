const TOKEN_KEY = "acme.vpToken";

/** A VP token, kept for this tab only, so closing it signs the VP out. */
export function storedToken(): string {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function storeToken(token: string): void {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    // Private windows may refuse storage; the token then lasts for this card.
  }
}
