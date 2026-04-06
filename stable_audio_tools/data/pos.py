import numpy as np
import re


class CondPosWave:
    @staticmethod
    def _position_map(position_str, motion_type):
        angle_map = {"left": 1, "left_front": 2, "front": 3, "right_front": 4, "right": 5}
        depth_map = {"near": 1, "medium": 2, "far": 3}

        position_str = position_str.lower()

        if motion_type == 'static' or 'static' in position_str:
            pattern = r"([^/]+)!([^/]+)\.wav$"
            regex = re.compile(pattern)
            match = regex.search(position_str)
            if match:
                angle = match.group(1)
                depth = match.group(2)
                return [angle_map[angle], depth_map[depth]], [angle_map[angle], depth_map[depth]]
            else:
                raise ValueError(f"position_str 不符合 static 格式: {position_str}")

        elif motion_type == 'dynamic':
            pattern = r"([^/]+)!([^/]+)2([^/]+)!([^/]+)\.wav$"
            regex = re.compile(pattern)
            match = regex.search(position_str)
            if match:
                direction1, depth1, direction2, depth2 = match.groups()
                return [angle_map[direction1], depth_map[depth1]], [angle_map[direction2], depth_map[depth2]]
            else:
                raise ValueError(f"position_str 不符合 dynamic 格式: {position_str}")
        else:
            raise ValueError(f"motion_type 无效: {motion_type}")

    @staticmethod
    def _resample_points(points, ori_sr, target_sr):
        """将原始采样点索引按采样率映射到目标采样率下"""
        scale = target_sr / ori_sr
        return [int(round(p * scale)) for p in points]

    @staticmethod
    def _generate_position_matrix_wave(
        from_pos, to_pos,
        start_samples, end_samples,
        num_samples,
    ):
        pos_matrix = np.zeros((num_samples, 3), dtype=np.float32)
        mask = np.zeros(num_samples, dtype=bool)

        start_samples = [max(0, min(i, num_samples - 1)) for i in start_samples]
        end_samples = [max(0, min(i, num_samples - 1)) for i in end_samples]

        min_s, max_s = min(start_samples), max(end_samples)

        if min_s < max_s:
            seg_len = max_s - min_s + 1
            t = np.linspace(0, 1, seg_len)
            for i in range(2):
                pos_matrix[min_s:max_s + 1, i] = from_pos[i] + t * (to_pos[i] - from_pos[i])

        for s, e in zip(start_samples, end_samples):
            if s < e:
                mask[s:e + 1] = True

        pos_matrix[mask, 2] = 1
        return pos_matrix

    @staticmethod
    def conditioning_pos_wave(
        movement_str,
        motion_type,
        start_samples,
        end_samples,
        num_samples_target,
        ori_sr,
        target_sr,
    ):
        """
        输入:
            movement_str: '(Distant Focus)right!far.wav'
            motion_type: 'static' / 'dynamic'
            start_samples, end_samples: 基于 ori_sr 的采样点
            num_samples_target: 重采样后音频的采样点数
            ori_sr, target_sr: 采样率映射
        输出:
            pos_matrix: [num_samples_target, 3]
        """
        from_pos, to_pos = CondPosWave._position_map(movement_str, motion_type)

        start_resampled = CondPosWave._resample_points(start_samples, ori_sr, target_sr)
        end_resampled = CondPosWave._resample_points(end_samples, ori_sr, target_sr)

        pos_matrix = CondPosWave._generate_position_matrix_wave(
            from_pos, to_pos, start_resampled, end_resampled, num_samples_target
        )

        return pos_matrix
