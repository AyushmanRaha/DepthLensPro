#!/usr/bin/env node
const { evaluateTarget } = require("../src/platform-support");
const arch = process.argv[2] || process.arch;
const platform = process.argv[3] || process.platform;
const target = evaluateTarget(platform, arch);
console.error(target.reason || `Unsupported native target ${target.label}.`);
process.exit(1);
