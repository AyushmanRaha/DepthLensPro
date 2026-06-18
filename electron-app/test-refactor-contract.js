const assert = require('assert');
const fs = require('fs');
const path = require('path');

const pkg = JSON.parse(fs.readFileSync(path.join(__dirname, 'package.json'), 'utf8'));
const scripts = pkg.scripts || {};

function assertNpmTestIncludesContractSuites() {
  const test = scripts.test || '';
  for (const suite of [
    'test-security-policy.js',
    'test-verify-resources.js',
    'test-platform-targets.js',
    'test-persistence-schema.js',
  ]) {
    assert(test.includes(suite), `npm test must include ${suite}`);
  }
}

function assertBuildScriptWrapsRawBuild(name, rawName, packagedName) {
  const command = scripts[name] || '';
  assert(command.includes('verify:resources'), `${name} must verify resources before raw build`);
  assert(command.includes(rawName), `${name} must run ${rawName}`);
  assert(command.includes(packagedName), `${name} must verify packaged resources after raw build`);
  assert(command.indexOf('verify:resources') < command.indexOf(rawName), `${name} must verify resources before ${rawName}`);
  assert(command.indexOf(rawName) < command.indexOf(packagedName), `${name} must run ${packagedName} after ${rawName}`);
}

assertNpmTestIncludesContractSuites();
for (const [name, rawName, packagedName] of [
  ['build:mac:arm64', 'build:mac:arm64:raw', 'verify:packaged:mac'],
  ['build:win:arm64', 'build:win:arm64:raw', 'verify:packaged:win'],
  ['build:win:x64', 'build:win:x64:raw', 'verify:packaged:win:x64'],
  ['build:linux:arm64', 'build:linux:arm64:raw', 'verify:packaged:linux'],
  ['build:linux:x64', 'build:linux:x64:raw', 'verify:packaged:linux:x64'],
]) {
  assertBuildScriptWrapsRawBuild(name, rawName, packagedName);
}

console.log('refactor contract tests passed');
