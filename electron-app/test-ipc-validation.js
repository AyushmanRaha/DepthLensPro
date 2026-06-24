"use strict";
const assert = require("assert");
const { validateDialogOptions, validateSettingsPayload } = require("./src/main/ipc-validation");
assert.deepStrictEqual(validateDialogOptions("open", { properties:["openFile"], filters:[{name:"Images", extensions:["png","jpg"]}] }).properties, ["openFile"]);
assert.throws(() => validateDialogOptions("open", { properties:["openFile"], extra:true }), /not allowed/);
assert.throws(() => validateDialogOptions("save", { filters:[{name:"Any", extensions:["exe"]}] }), /not allowed/);
assert.strictEqual(validateSettingsPayload({ theme:"dark" }).theme, "dark");
assert.throws(() => validateSettingsPayload(null), /object/);
console.log("IPC validation contract passed.");
