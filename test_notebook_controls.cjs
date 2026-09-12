// Exercise the notebook's actual browser-control code with a simulated DOM.
// This checks JS behavior, not Colab's permission UI or browser paint timing.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const notebook = JSON.parse(fs.readFileSync(path.join(__dirname, 'test_l1.ipynb'), 'utf8'));
const live = notebook.cells.find(c => c.metadata.tags?.includes('live')).source.join('');
const source = live.match(/display\(Javascript\(r"""([\s\S]*?)"""\)\)/)[1];
const elements = [];
const body = {append() {}};
function createElement(tag) {
    const element = {
        tag, style: {}, children: [], removed: false,
        append(...items) { this.children.push(...items); },
        remove() { this.removed = true; },
        setAttribute(key, value) { this[key] = value; },
        videoWidth: 1280, videoHeight: 720,
        readyState: 2, currentTime: 0,
        async play() {},
        getContext() { return {drawImage() {}, beginPath() {}, arc() {}, fill() {}}; },
        toDataURL(type, quality) { this.encoded = {type, quality}; return 'data:image/jpeg;base64,cGl4ZWxz'; },
    };
    elements.push(element);
    return element;
}
let track;
function makeStream() {
    track = {readyState: 'live', stop() { this.readyState = 'ended'; }};
    const current = track;
    return {getTracks: () => [current], getVideoTracks: () => [current]};
}
const mediaDevices = {getUserMedia: async () => makeStream()};
let now = 100, rafId = 0;
const rafs = new Map();
const context = {
    window: {}, document: {body, createElement}, navigator: {mediaDevices},
    performance: {now: () => now},
    requestAnimationFrame(fn) { rafs.set(++rafId, fn); return rafId; },
    cancelAnimationFrame(id) { rafs.delete(id); },
};
vm.runInNewContext(source, context);
const api = context.window;
let loads = 0, closes = 0, detections = 0;
api.l3LoadHandTracker = async () => {
    loads++;
    return {close() { closes++; }, detectForVideo() { detections++; return {landmarks: []}; }};
};
const flush = () => new Promise(resolve => setImmediate(resolve));
async function main() {
    assert.equal(await api.l1StartCamera(), true);
    const camera = api.l1Camera;
    assert.equal(camera.canvas.width, 480);
    assert.equal(camera.canvas.height, 270);
    assert.equal(camera.mode, 'manual');
    assert.equal(loads, 0); // Level 2 must not download/run the hand model.
    const sliders = elements.filter(e => e.type === 'range');
    assert.equal(sliders.length, 5);
    const packet = api.l1CaptureFrame();
    assert.equal(packet.light.x, -0.6);
    assert.equal(camera.canvas.encoded.quality, 0.75);
    camera.video.currentTime = 0.033;
    now += 33;
    const fresh = api.l1CaptureFrame();
    assert.equal(fresh.hand.frame_id, packet.hand.frame_id + 1);
    for (const [index, value] of [-1, 0.8, 0.5, 6, 0].entries()) {
        sliders[index].value = String(value);
        sliders[index].oninput();
    }
    const moved = api.l1CaptureFrame();
    assert.deepEqual(JSON.parse(JSON.stringify(moved.light)), {x: -1, y: 0.8, z: 0.5, power: 6, specular: 0});
    assert.equal(packet.light.x, -0.6); // Each capture owns a settings snapshot.
    elements.find(e => e.textContent === 'Reset light').onclick();
    assert.equal(api.l1CaptureFrame().light.power, 4);
    assert.equal(api.l1CaptureFrame().light.x, -0.6);
    // The same live checkbox lazily restores Level 3, and manual mode stops inference.
    const mode = elements.find(e => e.type === 'checkbox');
    mode.checked = true;
    mode.onchange();
    await flush();
    assert.equal(loads, 1);
    camera.video.currentTime += 0.05;
    camera.updateFrame(now += 50);
    assert.equal(detections, 1);
    mode.checked = false;
    mode.onchange();
    camera.video.currentTime += 0.05;
    camera.updateFrame(now += 50);
    assert.equal(detections, 1);
    elements.find(e => e.textContent === 'Stop camera').onclick();
    assert.equal(track.readyState, 'ended');
    assert.equal(camera.wrapper.removed, true);
    assert.equal(api.l1CaptureFrame(), null);
    assert.equal(closes, 1);
    assert.equal(rafs.size, 0);

    // Smaller capture preserves a wide camera's aspect ratio and JPEG preference.
    assert.equal(await api.l1StartCamera({width: 320, quality: 0.6}), true);
    assert.equal(api.l1Camera.canvas.width, 320);
    assert.equal(api.l1Camera.canvas.height, 180);
    api.l1CaptureFrame();
    assert.equal(api.l1Camera.canvas.encoded.quality, 0.6);
    api.l1StopCamera();

    // Stop while camera permission is pending must release the eventual stream.
    let resolve;
    mediaDevices.getUserMedia = () => new Promise(r => { resolve = r; });
    const pending = api.l1StartCamera();
    // A checkbox click while awaiting permission must not hit a temporal-dead-zone error.
    const pendingMode = elements.filter(e => e.type === 'checkbox').at(-1);
    pendingMode.checked = true;
    pendingMode.onchange();
    api.l1StopCamera();
    resolve(makeStream());
    assert.equal(await pending, false);
    await flush();
    assert.equal(track.readyState, 'ended');

    mediaDevices.getUserMedia = async () => { throw new Error('permission denied'); };
    await assert.rejects(api.l1StartCamera(), /permission denied/);
    assert.equal(api.l1CaptureFrame(), null);
    console.log('Notebook JS checks passed: lazy hand tracking, manual fresh capture, resolution/quality, controls, snapshots, stop, permission race, cleanup.');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
