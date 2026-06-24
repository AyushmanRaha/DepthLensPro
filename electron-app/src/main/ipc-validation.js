"use strict";
const path = require("path");
const ALLOWED_OPEN_PROPS = new Set(["title", "defaultPath", "buttonLabel", "filters", "properties", "message", "securityScopedBookmarks"]);
const ALLOWED_SAVE_PROPS = new Set(["title", "defaultPath", "buttonLabel", "filters", "message", "nameFieldLabel", "showsTagField"]);
const ALLOWED_OPEN_PROPERTIES = new Set(["openFile", "openDirectory", "multiSelections", "showHiddenFiles", "createDirectory", "promptToCreate", "noResolveAliases", "treatPackageAsDirectory", "dontAddToRecent"]);
const ALLOWED_EXTENSIONS = new Set(["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff", "npy", "ply", "obj", "json", "txt", "csv"]);
function plainObject(value) { return value && typeof value === "object" && !Array.isArray(value) && Object.getPrototypeOf(value) === Object.prototype; }
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
const ALLOWED_SETTINGS_KEYS = new Set(["schemaVersion", "selectedModel", "selectedDevice", "selectedColormap", "targetFps", "webcamFrameMaxDimension", "smoothingPreference", "recentBenchmarkSettings", "onnxPreference", "onnxStatus", "ui", "privacy"]);
const ALLOWED_TARGET_FPS = new Set([1, 2, 3, 5]);
const ALLOWED_FRAME_DIMS = new Set([256, 384, 512]);
const ALLOWED_ONNX_PREFERENCE = new Set(["auto", "enabled", "disabled"]);
const ALLOWED_ONNX_STATUS = new Set(["unknown", "available", "unavailable", "disabled", "error"]);
function rejectDangerousValue(value, label) {
  if (typeof value === "function" || typeof value === "symbol") throw new Error(`${label} contains unsupported value`);
  if (Array.isArray(value)) throw new Error(`${label} must not be an array`);
  if (value && typeof value === "object" && !plainObject(value)) throw new Error(`${label} must be a plain object`);
}
function validateFiniteNumber(value, label, min, max) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) throw new Error(`${label} must be a number between ${min} and ${max}`);
  return value;
}
function validateString(value, label, max = 80) {
  if (typeof value !== "string" || value.includes("\0") || value.length > max) throw new Error(`${label} must be a safe string`);
  return value;
}
function validateSettingsPayload(payload) {
  if (!plainObject(payload)) throw new Error("settings payload must be an object");
  const out = {};
  for (const [key, value] of Object.entries(payload)) {
    if (!ALLOWED_SETTINGS_KEYS.has(key)) throw new Error(`settings key not allowed: ${key}`);
    rejectDangerousValue(value, `settings.${key}`);
    if (key === "schemaVersion") {
      if (value !== 1) throw new Error("settings.schemaVersion is unsupported");
      out[key] = value;
    } else if (["selectedModel", "selectedDevice", "selectedColormap"].includes(key)) {
      out[key] = validateString(value, `settings.${key}`);
    } else if (key === "targetFps") {
      if (!ALLOWED_TARGET_FPS.has(value)) throw new Error("settings.targetFps is not allowed");
      out[key] = value;
    } else if (key === "webcamFrameMaxDimension") {
      if (!ALLOWED_FRAME_DIMS.has(value)) throw new Error("settings.webcamFrameMaxDimension is not allowed");
      out[key] = value;
    } else if (key === "smoothingPreference") {
      out[key] = validateFiniteNumber(value, "settings.smoothingPreference", 0, 0.95);
    } else if (key === "recentBenchmarkSettings") {
      if (!plainObject(value)) throw new Error("settings.recentBenchmarkSettings must be an object");
      const allowed = new Set(["model", "device", "iterations"]);
      for (const nestedKey of Object.keys(value)) if (!allowed.has(nestedKey)) throw new Error(`settings.recentBenchmarkSettings key not allowed: ${nestedKey}`);
      out[key] = {};
      if (value.model !== undefined) out[key].model = validateString(value.model, "settings.recentBenchmarkSettings.model");
      if (value.device !== undefined) out[key].device = validateString(value.device, "settings.recentBenchmarkSettings.device");
      if (value.iterations !== undefined) {
        if (![1, 2, 3, 5, 10].includes(value.iterations)) throw new Error("settings.recentBenchmarkSettings.iterations is not allowed");
        out[key].iterations = value.iterations;
      }
    } else if (key === "onnxPreference") {
      if (!ALLOWED_ONNX_PREFERENCE.has(value)) throw new Error("settings.onnxPreference is not allowed");
      out[key] = value;
    } else if (key === "onnxStatus") {
      if (!ALLOWED_ONNX_STATUS.has(value)) throw new Error("settings.onnxStatus is not allowed");
      out[key] = value;
    } else if (["ui", "privacy"].includes(key)) {
      if (!plainObject(value)) throw new Error(`settings.${key} must be an object`);
      for (const nestedValue of Object.values(value)) {
        if (nestedValue && typeof nestedValue === "object") throw new Error(`settings.${key} must not contain nested objects`);
        rejectDangerousValue(nestedValue, `settings.${key}`);
      }
      out[key] = value;
    }
  }
  return out;
}
module.exports = { validateDialogOptions, validateSettingsPayload, ALLOWED_EXTENSIONS, ALLOWED_SETTINGS_KEYS };
