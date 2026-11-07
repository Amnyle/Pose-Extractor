// Write android/app/src/test/resources/golden.txt for PoseLogicTest (plain text, no JSON parser
// needed in the test): the reference routines of app/tests/golden.json with the Python tool's labels,
// holds and waists, and the app's finishedPoses after every second.
//   node mobile/scripts/make-java-golden.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { buildSample, finishedPoses } from '../../app/js/pipeline.js';

const golden = JSON.parse(readFileSync(new URL('../../app/tests/golden.json', import.meta.url)));
const out = [];
for (const g of golden) {
  out.push(`routine ${g.seed} ${g.input.length}`);
  for (const [i, s] of g.input.entries()) {
    const e = g.expected.samples[i];
    out.push(['frame', s.time, s.cut ? 1 : 0, e.label ?? '-', e.waist ?? '-', s.maskRect ? s.maskRect.join(',') : '-',
      s.width, s.height, s.kp.flat().join(',')].join(' '));
  }
  for (const [a, b, label] of g.expected.holds) out.push(`hold ${a} ${b} ${label ?? '-'}`);
  const samples = g.input.map((s) => buildSample(s));
  for (let n = 6; n <= samples.length; n += 6) out.push(`finished ${n} ${[...finishedPoses(samples.slice(0, n))].sort().join(',') || '-'}`);
}
writeFileSync(new URL('../android/app/src/test/resources/golden.txt', import.meta.url), out.join('\n') + '\n');
console.log(out.length, 'lines');
