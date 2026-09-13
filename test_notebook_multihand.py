"""Synthetic independent-hand tracking and linear multi-light accumulation checks."""
import ast
import numpy as np
import torch
from test_notebook_performance import load_helpers
from test_notebook_gestures import finger_hand


def hand(x, ratio):
    points = finger_hand(ratio)
    for point in points:
        point.x += x - .5
    return points


def main():
    nb, ns = load_helpers()
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(line for line in cell['source'] if not line.lstrip().startswith('%')))
    manager = ns['MultiPalmLights']()
    seed = ns['default_light']()
    hands = [hand(x, ratio) for x, ratio in ((.2, .7), (.4, 1.8), (.6, .7), (.8, 1.8))]
    for i in range(30):
        lights, landmarks, ids, _ = manager.update(hands if i % 2 else hands[::-1], 640, 480, i*.05, seed)
    assert len(lights) == len(landmarks) == len(set(ids)) == 4
    ordered = sorted(lights, key=lambda light: light['x'])
    for light, points in zip(lights, landmarks):
        assert abs(light['screen_u'] - np.mean([points[i].x for i in (0, 5, 9, 13, 17)])) < 1e-7
        assert abs(light['screen_v'] - np.mean([points[i].y for i in (0, 5, 9, 13, 17)])) < 1e-7
    assert ordered[0]['power'] < .1 and ordered[1]['power'] > 11.9
    assert ordered[2]['power'] < .1 and ordered[3]['power'] > 11.9
    previous = {track['id']: dict(track['light']) for track in manager.tracks}
    manager.update(hands[::-1], 640, 480, 1.5, seed)
    for track in manager.tracks:
        assert abs(track['light']['x'] - previous[track['id']]['x']) < .001
    snapshot = [dict(light) for light in lights]
    faded, _, _, _ = manager.update([], 640, 480, 1.9, seed)
    assert max(light['power'] for light in faded) < 6.1
    assert manager.update([], 640, 480, 2.2, seed)[0] == []
    assert lights == snapshot
    assert manager.update([[object()]], 640, 480, 2.3, seed)[0] == []

    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        estimator = ns['ScharrNormalEstimator'](25, 33, device=device)
        depth = torch.full((25, 33), 2., device=device)
        n, p, valid = estimator.compute_normals_and_coords(depth)
        frame = np.full((25, 33, 3), 70, np.uint8)
        lamp = dict(seed, power=.4)
        render = ns['relight_rgb']
        # Two identical lights equal one double-power light, including one ambient term.
        for radius in (0., .35):
            opts = dict(cull_radius_frac=radius, flat_threshold=.95)
            actual = render(frame, n, p, valid, [lamp, lamp], **opts)
            expected = render(frame, n, p, valid, dict(lamp, power=.8), **opts)
            np.testing.assert_array_equal(actual, expected)
            ones, zeros = torch.ones_like(depth), torch.zeros_like(depth)
            actual = render(frame, n, p, valid, [lamp, lamp], visibility=[ones, zeros], **opts)
            np.testing.assert_array_equal(actual, render(frame, n, p, valid, lamp, **opts))
        np.testing.assert_array_equal(render(frame, n, p, valid, []),
                                      render(frame, n, p, valid, dict(lamp, power=0)))
        # Hand footprint follows measured screen coordinates, independently of XYZ lag.
        h, w = 81, 121
        estimator = ns['ScharrNormalEstimator'](h, w, device=device)
        n2, p2, v2 = estimator.compute_normals_and_coords(torch.full((h, w), 2., device=device))
        frame2 = np.full((h, w, 3), 100, np.uint8)
        ambient = render(frame2, n2, p2, v2, [])
        yy, xx = np.mgrid[:h, :w]
        lamps = [dict(lamp, power=1., screen_u=u, screen_v=.5) for u in (.25, .75)]
        total_mask = np.zeros((h, w), bool)
        for value in lamps:
            cx, cy = ns['_light_screen_pixels'](value, h, w)
            assert cx == round(value['screen_u']*(w-1))
            radius = np.hypot(xx-cx, yy-cy)
            result = render(frame2, n2, p2, v2, value, cull_radius_frac=.2)
            np.testing.assert_array_equal(result[radius >= .2*w], ambient[radius >= .2*w])
            full = render(frame2, n2, p2, v2, value)
            np.testing.assert_array_equal(result[radius <= .65*.2*w], full[radius <= .65*.2*w])
            rim = (radius > .75*.2*w) & (radius < .9*.2*w)
            assert np.any(result[rim] < full[rim]) and np.any(result[rim] > ambient[rim])
            total_mask |= radius < .2*w
        both = render(frame2, n2, p2, v2, lamps, cull_radius_frac=.2)
        np.testing.assert_array_equal(both[~total_mask], ambient[~total_mask])
        try:
            render(frame, n, p, valid, [lamp, lamp], visibility=[ones])
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched shadow maps accepted')
    print('Multi-hand checks passed: four lights, reordered detections, independent power, loss/fade, linear sum, per-light shadows, ambient once, CPU/CUDA where available.')


if __name__ == '__main__':
    main()
