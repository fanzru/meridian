import { Connection } from "@solana/web3.js";

function nonEmptyString(...values) {
  for (const value of values) {
    if (typeof value !== "string") continue;
    const trimmed = value.trim();
    if (trimmed) return trimmed;
  }
  return null;
}

function normalizeProviderName(value, fallback = "alchemy") {
  const name = String(value ?? fallback).trim().toLowerCase();
  if (name === "helius" || name === "alchemy") return name;
  return fallback;
}

function resolveRpcUrl(scope, fallbackUrl = null) {
  if (scope === "pnl") {
    return nonEmptyString(process.env.PNL_RPC_URL, fallbackUrl, process.env.RPC_URL);
  }
  return nonEmptyString(process.env.RPC_URL, fallbackUrl);
}

function createProvider(name, label) {
  const connections = new Map();

  return {
    name,
    label,
    resolveUrl(scope = "main", fallbackUrl = null) {
      return resolveRpcUrl(scope, fallbackUrl);
    },
    getConnection(scope = "main", fallbackUrl = null) {
      const url = this.resolveUrl(scope, fallbackUrl);
      if (!url) {
        throw new Error(`RPC URL not set for ${scope} (${label})`);
      }
      const cacheKey = `${scope}:${url}`;
      if (!connections.has(cacheKey)) {
        connections.set(cacheKey, new Connection(url, "confirmed"));
      }
      return connections.get(cacheKey);
    },
  };
}

export const rpcProviders = {
  alchemy: createProvider("alchemy", "Alchemy"),
  helius: createProvider("helius", "Helius"),
};

export function getRpcProvider(scope = "main") {
  const providerName = scope === "pnl"
    ? normalizeProviderName(process.env.PNL_RPC_PROVIDER)
    : normalizeProviderName(process.env.RPC_PROVIDER);
  return rpcProviders[providerName] ?? rpcProviders.alchemy;
}

export function getRpcConnection(scope = "main", fallbackUrl = null) {
  return getRpcProvider(scope).getConnection(scope, fallbackUrl);
}
