"use strict";
const path = require("path");
const ALLOWED_OPEN_PROPS = new Set(["title", "defaultPath", "buttonLabel", "filters", "properties", "message", "securityScopedBookmarks"]);
const ALLOWED_SAVE_PROPS = new Set(["title", "defaultPath", "buttonLabel", "filters", "message", "nameFieldLabel", "showsTagField"]);
const ALLOWED_OPEN_PROPERTIES = new Set(["openFile", "openDirectory", "multiSelections", "showHiddenFiles", "createDirectory", "promptToCreate", "noResolveAliases", "treatPackageAsDirectory", "dontAddToRecent"]);
const ALLOWED_EXTENSIONS = new Set(["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff", "npy", "ply", "obj", "json", "txt", "csv"]);
function plainObject(value) { return value && typeof value === "object" && !Array.isArray(value); }
function validateFilters(filters) {
  if (filters === undefined) return undefined;
  if (!Array.isArray(filters)) throw new Error("dialog filters must be an array");
  return filters.map((f) => {
    if (!plainObject(f) || typeof f.name !== "string" || !Array.isArray(f.extensions)) throw new Error("invalid dialog filter");
    const extensions = f.extensions.map((ext) => String(ext).toLowerCase().replace(/^\./, ""));
    if (!extensions.every((ext) => ALLOWED_EXTENSIONS.has(ext))) throw new Error("dialog filter extension is not allowed");
    return { name: f.name.slice(0, 80), extensions };
  });
}
function safeDefaultPath(value) {
  if (value === undefined) return undefined;
  if (typeof value !== "string") throw new Error("defaultPath must be a string");
  if (value.includes("\0")) throw new Error("defaultPath contains invalid characters");
  return path.normalize(value);
}
function validateDialogOptions(kind, options = {}) {
  if (!plainObject(options)) throw new Error("dialog options must be an object");
  const allowed = kind === "open" ? ALLOWED_OPEN_PROPS : ALLOWED_SAVE_PROPS;
  for (const key of Object.keys(options)) if (!allowed.has(key)) throw new Error(`dialog option not allowed: ${key}`);
  const out = {};
  for (const key of ["title", "buttonLabel", "message", "nameFieldLabel"]) if (options[key] !== undefined) out[key] = String(options[key]).slice(0, 160);
  if (options.defaultPath !== undefined) out.defaultPath = safeDefaultPath(options.defaultPath);
  if (options.filters !== undefined) out.filters = validateFilters(options.filters);
  if (kind === "open" && options.properties !== undefined) {
    if (!Array.isArray(options.properties)) throw new Error("dialog properties must be an array");
    out.properties = options.properties.map(String).filter((p) => ALLOWED_OPEN_PROPERTIES.has(p));
    if (!out.properties.length) out.properties = ["openFile"];
  }
  return out;
}
function validateSettingsPayload(payload) {
  if (!plainObject(payload)) throw new Error("settings payload must be an object");
  return payload;
}
module.exports = { validateDialogOptions, validateSettingsPayload, ALLOWED_EXTENSIONS };
