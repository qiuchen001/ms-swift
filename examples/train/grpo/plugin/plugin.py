import asyncio
import os
import re
import textwrap
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Optional

import json
import torch

from swift.llm import PtEngine, RequestConfig, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse
from swift.plugin import ORM, orms, rm_plugins
# register context manager(used in gym training)
from swift.plugin.context_manager import ContextManager, context_managers
from swift.plugin.env import Env, envs
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()
"""
TO CUSTOMIZE REWARD FUNCTION:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

    Step 2: Add your reward function to the orms registry:
        orms['my_reward_function'] = MyRewardFunction

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_funcs my_reward_function
"""


# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
        return rewards


class MathFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """

        logger.info(f"MultiModalAccuracyORM completions: {completions}")
        logger.info(f"MultiModalAccuracyORM solution: {solution}")

        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


class MultiModalAccuracyClassificationORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """

        logger.info(f"MultiModalAccuracyClassificationORM completions: {completions}")
        logger.info(f"MultiModalAccuracyClassificationORM solution: {solution}")

        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    # 比较第一个字符是否相同（同时确保字符串非空）
                    if student_answer and ground_truth and student_answer[0] == ground_truth[0]:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


class VideoCategoryIoUReward(ORM):
    """
    奖励函数，处理的模型输出和目标答案格式为 ```json``` 代码块：
    ```json
    [
      {
        "category": "白天",
        "start_time_seconds": 0.0,
        "end_time_seconds": 16.0
      },
      ...
    ]
    ```
    1. 从代码块中提取JSON内容
    2. 检查是否为合法JSON格式
    3. 检查category与目标是否匹配
    4. 匹配时计算时间区间的IoU作为奖励
    """

    def __call__(self, completions, solution, **kwargs):
        """
        Args:
            completions: list[str]，模型输出
            solution: list[str]，目标答案（同样格式）
        Returns:
            list[float]: 每个样本的奖励分数
        """
        logger.info(f"VideoCategoryIoUReward completions: {completions}")
        logger.info(f"VideoCategoryIoUReward solution: {solution}")

        rewards = []
        for pred, gt in zip(completions, solution):
            # 1. 提取```json```代码块中的内容
            match = re.search(r"```json\s*([\s\S]*?)\s*```", pred, re.DOTALL)
            if not match:
                rewards.append(0.0)
                continue
            pred_json_str = match.group(1)
            # 2. 检查是否为合法JSON
            try:
                pred_list = json.loads(pred_json_str)
                assert isinstance(pred_list, list)
            except Exception:
                rewards.append(0.0)
                continue
            # 3. 解析目标，提取```json```代码块中的内容
            try:
                gt_match = re.search(r"```json\s*([\s\S]*?)\s*```", gt, re.DOTALL)
                if not gt_match:
                    rewards.append(0.0)
                    continue
                gt_json_str = gt_match.group(1)
                gt_list = json.loads(gt_json_str)
                assert isinstance(gt_list, list)
            except Exception:
                rewards.append(0.0)
                continue

            # 4. category匹配与IoU奖励
            total_reward = 0.0
            match_count = 0
            for gt_item in gt_list:
                gt_cat = gt_item["category"]
                gt_start = float(gt_item["start_time_seconds"])
                gt_end = float(gt_item["end_time_seconds"])
                # 找到所有预测中同类category
                pred_items = [item for item in pred_list if item.get("category") == gt_cat]
                if not pred_items:
                    continue
                # 计算IoU，取最大IoU
                best_iou = 0.0
                for pred_item in pred_items:
                    try:
                        p_start = float(pred_item["start_time_seconds"])
                        p_end = float(pred_item["end_time_seconds"])
                        inter = max(0.0, min(gt_end, p_end) - max(gt_start, p_start))
                        union = max(gt_end, p_end) - min(gt_start, p_start)
                        iou = inter / union if union > 0 else 0.0
                        if iou > best_iou:
                            best_iou = iou
                    except Exception:
                        continue
                total_reward += best_iou
                match_count += 1
            # 平均IoU作为奖励（如果没有匹配则为0）
            reward = total_reward / match_count if match_count > 0 else 0.0
            rewards.append(reward)
        return rewards


class VideoCategoryIoUReward_v2(ORM):
    """
    奖励函数，处理的模型输出和目标答案格式为 ```json``` 代码块：
    ```json
    [
      {
        "category": "白天",
        "start_time_seconds": 0.0,
        "end_time_seconds": 16.0
      },
      ...
    ]
    ```
    1. 从代码块中提取JSON内容
    2. 检查是否为合法JSON格式
    3. 检查category与目标是否匹配
    4. 匹配时计算时间区间的IoU作为奖励
    """

    def __call__(self, completions, solution, **kwargs):
        """
        Args:
            completions: list[str]，模型输出
            solution: list[str]，目标答案（同样格式）
        Returns:
            list[float]: 每个样本的奖励分数
        """

        logger.info(f"VideoCategoryIoUReward completions: {completions}")
        logger.info(f"VideoCategoryIoUReward solution: {solution}")

        rewards = []
        for pred, gt in zip(completions, solution):
            # 1. 提取```json```代码块中的内容
            match = re.search(r"```json\s*([\s\S]*?)\s*```", pred, re.DOTALL)
            if not match:
                rewards.append(0.0)
                continue
            pred_json_str = match.group(1)
            # 2. 检查是否为合法JSON
            try:
                pred_list = json.loads(pred_json_str)
                assert isinstance(pred_list, list)
            except Exception:
                rewards.append(0.0)
                continue
            # 3. 解析目标，提取```json```代码块中的内容
            try:
                gt_match = re.search(r"```json\s*([\s\S]*?)\s*```", gt, re.DOTALL)
                if not gt_match:
                    rewards.append(0.0)
                    continue
                gt_json_str = gt_match.group(1)
                gt_list = json.loads(gt_json_str)
                assert isinstance(gt_list, list)
            except Exception:
                rewards.append(0.0)
                continue

            # 4. 优化后的奖励计算逻辑：同分类多时间段一一最优配对
            total_iou_reward = 0.0
            matched_gt_count = 0
            matched_pred_indices = set()
            matched_gt_indices = set()

            # 按分类分组
            from collections import defaultdict
            gt_by_cat = defaultdict(list)
            pred_by_cat = defaultdict(list)
            for idx, item in enumerate(gt_list):
                gt_by_cat[item["category"]].append((idx, item))
            for idx, item in enumerate(pred_list):
                pred_by_cat[item.get("category")].append((idx, item))

            for cat in gt_by_cat:
                gt_items = gt_by_cat[cat]
                pred_items = pred_by_cat.get(cat, [])
                # 构建IoU矩阵
                iou_matrix = []
                for gt_idx, gt_item in gt_items:
                    row = []
                    gt_start = float(gt_item["start_time_seconds"])
                    gt_end = float(gt_item["end_time_seconds"])
                    for pred_idx, pred_item in pred_items:
                        try:
                            p_start = float(pred_item["start_time_seconds"])
                            p_end = float(pred_item["end_time_seconds"])
                            inter = max(0.0, min(gt_end, p_end) - max(gt_start, p_start))
                            union = max(gt_end, p_end) - min(gt_start, p_start)
                            iou = inter / union if union > 0 else 0.0
                        except Exception:
                            iou = 0.0
                        row.append((iou, gt_idx, pred_idx))
                    iou_matrix.append(row)
                # 贪心配对：每次选最大IoU且未被配对的
                pairs = []
                used_gt = set()
                used_pred = set()
                all_pairs = []
                for row in iou_matrix:
                    all_pairs.extend(row)
                all_pairs.sort(reverse=True, key=lambda x: x[0])  # 按IoU降序
                for iou, gt_idx, pred_idx in all_pairs:
                    if iou == 0.0:
                        continue
                    if gt_idx in used_gt or pred_idx in used_pred:
                        continue
                    pairs.append((gt_idx, pred_idx, iou))
                    used_gt.add(gt_idx)
                    used_pred.add(pred_idx)
                # 统计配对
                total_iou_reward += sum(iou for _, _, iou in pairs)
                matched_gt_count += len(pairs)
                matched_gt_indices.update(gt_idx for gt_idx, _, _ in pairs)
                matched_pred_indices.update(pred_idx for _, pred_idx, _ in pairs)

            # 平均IoU作为基础奖励（如果没有匹配则为0）
            avg_iou_reward = total_iou_reward / len(gt_list) if gt_list else 0.0

            # 4.2 计算漏报惩罚 (False Negative)
            # 新逻辑：只要预测中有同分类，无论时间段是否重叠，都不算漏报
            fn_count = 0
            for cat in gt_by_cat:
                if not pred_by_cat.get(cat):
                    # 预测中没有该分类，才算漏报（该分类下所有gt item都算漏报）
                    fn_count += len(gt_by_cat[cat])
            fn_penalty = fn_count * 0.1

            # 4.3 计算误报惩罚 (False Positive)
            fp_count = len(pred_list) - len(matched_pred_indices)
            fp_penalty = fp_count * 0.1

            # 4.4 组合最终奖励
            final_reward = avg_iou_reward - fn_penalty - fp_penalty
            # final_reward = max(0.0, final_reward)
            rewards.append(final_reward)
        return rewards


class VideoCategoryIoUReward_v3(ORM):
    """
    改进的奖励函数，解决以下问题：
    1. 分离格式验证和内容评估
    2. 增强惩罚力度
    3. 确保奖励始终为正
    4. 添加更详细的调试信息
    """

    def __call__(self, completions, solution, **kwargs):
        """
        Args:
            completions: list[str]，模型输出
            solution: list[str]，目标答案
        Returns:
            list[float]: 每个样本的奖励分数
        """

        logger.info(f"VideoCategoryIoUReward_v3 completions: {completions}")
        logger.info(f"VideoCategoryIoUReward_v3 solution: {solution}")

        rewards = []
        for pred, gt in zip(completions, solution):
            reward = self._calculate_reward(pred, gt)
            rewards.append(reward)
            
        return rewards

    def _calculate_reward(self, pred, gt):
        """计算单个样本的奖励"""
        
        # 1. 格式验证阶段 (权重: 0.3)
        format_score = self._validate_format(pred)
        if format_score == 0:
            return 0.0  # 格式错误直接返回0
        
        # 2. 解析JSON内容
        pred_list = self._parse_json(pred)
        gt_list = self._parse_json(gt)
        
        if pred_list is None or gt_list is None:
            return 0.0
        
        # 3. 分类准确性评估 (权重: 0.4)
        classification_score = self._evaluate_classification(pred_list, gt_list)
        
        # 4. 时间覆盖度评估 (权重: 0.3)
        coverage_score = self._evaluate_coverage(pred_list, gt_list)
        
        # 5. 组合最终奖励 (确保为正)
        final_reward = (
            0.3 * format_score + 
            0.4 * classification_score + 
            0.3 * coverage_score
        )
        
        # 确保奖励在[0, 1]范围内
        final_reward = max(0.0, min(1.0, final_reward))
        
        logger.info(f"Reward breakdown: format={format_score:.3f}, "
                    f"classification={classification_score:.3f}, "
                    f"coverage={coverage_score:.3f}, "
                    f"final={final_reward:.3f}")
        
        return final_reward

    def _validate_format(self, text):
        """验证JSON格式 (0-1分)"""
        try:
            # 检查是否包含```json```代码块
            match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.DOTALL)
            if not match:
                return 0.0
            
            json_str = match.group(1)
            # 尝试解析JSON
            data = json.loads(json_str)
            if not isinstance(data, list):
                return 0.0
            
            # 检查每个元素的结构
            for item in data:
                if not isinstance(item, dict):
                    return 0.0
                required_keys = {"category", "start_time_seconds", "end_time_seconds"}
                if not required_keys.issubset(set(item.keys())):
                    return 0.0
                # 检查时间值是否为数字
                if not (isinstance(item["start_time_seconds"], (int, float)) and 
                       isinstance(item["end_time_seconds"], (int, float))):
                    return 0.0
                # 检查时间逻辑
                if item["start_time_seconds"] >= item["end_time_seconds"]:
                    return 0.0
            
            return 1.0
            
        except Exception:
            return 0.0

    def _parse_json(self, text):
        """解析JSON内容"""
        try:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.DOTALL)
            if not match:
                return None
            return json.loads(match.group(1))
        except Exception:
            return None

    def _evaluate_classification(self, pred_list, gt_list):
        """评估分类准确性 (0-1分)"""
        if not gt_list:
            return 1.0 if not pred_list else 0.0
        
        # 统计分类
        gt_categories = set(item["category"] for item in gt_list)
        pred_categories = set(item["category"] for item in pred_list)
        
        # 计算精确率和召回率
        if not pred_categories:
            precision = 0.0
            recall = 0.0
        else:
            precision = len(gt_categories & pred_categories) / len(pred_categories)
            recall = len(gt_categories & pred_categories) / len(gt_categories)
        
        # 使用F1分数
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return f1

    def _evaluate_coverage(self, pred_list, gt_list):
        """评估时间覆盖度 (0-1分)"""
        if not gt_list:
            return 1.0 if not pred_list else 0.0
        
        # 按分类分组
        from collections import defaultdict
        gt_by_cat = defaultdict(list)
        pred_by_cat = defaultdict(list)
        
        for item in gt_list:
            gt_by_cat[item["category"]].append(item)
        for item in pred_list:
            pred_by_cat[item["category"]].append(item)
        
        total_iou = 0.0
        matched_count = 0
        
        # 对每个分类计算IoU
        for cat in gt_by_cat:
            if cat not in pred_by_cat:
                continue
            
            gt_items = gt_by_cat[cat]
            pred_items = pred_by_cat[cat]
            
            # 计算最佳匹配的IoU
            best_iou = 0.0
            for gt_item in gt_items:
                gt_start = float(gt_item["start_time_seconds"])
                gt_end = float(gt_item["end_time_seconds"])
                
                for pred_item in pred_items:
                    try:
                        p_start = float(pred_item["start_time_seconds"])
                        p_end = float(pred_item["end_time_seconds"])
                        
                        # 计算IoU
                        inter_start = max(gt_start, p_start)
                        inter_end = min(gt_end, p_end)
                        
                        if inter_start < inter_end:
                            intersection = inter_end - inter_start
                            union = (gt_end - gt_start) + (p_end - p_start) - intersection
                            iou = intersection / union if union > 0 else 0.0
                            best_iou = max(best_iou, iou)
                    except Exception:
                        continue
            
            total_iou += best_iou
            matched_count += 1
        
        # 计算平均IoU
        avg_iou = total_iou / len(gt_by_cat) if gt_by_cat else 0.0
        
        return avg_iou


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        return rewards


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards


class CodeRewardByJudge0(ORM):
    LANGUAGE_ID_MAP = {
        'assembly': 45,
        'bash': 46,
        'basic': 47,
        'c': 50,
        'c++': 54,
        'clojure': 86,
        'c#': 51,
        'cobol': 77,
        'common lisp': 55,
        'd': 56,
        'elixir': 57,
        'erlang': 58,
        'executable': 44,
        'f#': 87,
        'fortran': 59,
        'go': 60,
        'groovy': 88,
        'haskell': 61,
        'java': 62,
        'javascript': 63,
        'kotlin': 78,
        'lua': 64,
        'multi-file program': 89,
        'objective-c': 79,
        'ocaml': 65,
        'octave': 66,
        'pascal': 67,
        'perl': 85,
        'php': 68,
        'plain text': 43,
        'prolog': 69,
        'python': 71,
        'python2': 70,
        'python3': 71,
        'r': 80,
        'ruby': 72,
        'rust': 73,
        'scala': 81,
        'sql': 82,
        'swift': 83,
        'typescript': 74,
        'visual basic.net': 84
    }
    PYTHON_ID = 71

    def __init__(self):
        self.endpoint = os.getenv('JUDGE0_ENDPOINT')
        assert self.endpoint is not None, (
            'Judge0 endpoint is not set. Please set the JUDGE0_ENDPOINT environment variable.')
        x_auth_token = os.getenv('JUDGE0_X_AUTH_TOKEN')
        self.headers = {'Content-Type': 'application/json'}
        if x_auth_token is not None:
            self.headers['X-Auth-Token'] = x_auth_token

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    @classmethod
    def get_language_id(cls, language):
        if language is None:
            return cls.PYTHON_ID
        return cls.LANGUAGE_ID_MAP.get(language.lower().strip(), cls.PYTHON_ID)

    async def _evaluate_code(self, code, test_cases, language_id):
        import aiohttp
        try:
            passed = 0
            total = len(test_cases)

            for case in test_cases:
                if code is not None and code != '':
                    async with aiohttp.ClientSession() as session:
                        payload = {
                            'source_code': code,
                            'language_id': language_id,
                            'stdin': case['input'],
                            'expected_output': case['output']
                        }
                        logger.debug(f'Payload: {payload}')
                        async with session.post(
                                self.endpoint + '/submissions/?wait=true', json=payload,
                                headers=self.headers) as response:
                            response_json = await response.json()
                            logger.debug(f'Response: {response_json}')
                            if response_json['status']['description'] == 'Accepted':
                                passed += 1

            success_rate = (passed / total)
            return success_rate
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            return 0.0

    def run_async_from_sync(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            rewards = loop.run_until_complete(self.run_async())
        finally:
            loop.close()
        return rewards

    async def run_async(self):
        tasks = [
            self._evaluate_code(code, info['test_cases'], CodeRewardByJudge0.get_language_id(info['language']))
            for code, info in zip(self.code_snippets, self.verification_info)
        ]
        results = await asyncio.gather(*tasks)
        rewards = list(results)
        return rewards

    def __call__(self, completions, **kwargs) -> List[float]:
        self.verification_info = kwargs['verification_info']

        languages = [info['language'] for info in self.verification_info]
        self.code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]

        try:
            rewards = self.run_async_from_sync()
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            rewards = [0.0] * len(completions)
        return rewards


# ref implementation: https://github.com/qiancheng0/ToolRL/blob/main/verl/utils/reward_score/rlla.py
# arxiv paper: https://arxiv.org/abs/2504.13958
# MAX1STEP30MAX3: enable Two stage reward Setting include Format and Correctness
# SCHEDULEREWARD: enable Dynamic (Finegrained) reward Setting include Format and Correctness
# Correctness Reward Granularity:
# COARSEREWARD -> Coarse, INTERMEDIATEREWARD -> Intermediate, REFINEDREWARD -> Finegrained
class ToolUseFormatReward(ORM):

    def __init__(self):
        self.format_max_possible = 1.0
        self.format_min_possible = 0.0

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        max_possible_reward = self.format_max_possible
        min_possible_reward = self.format_min_possible
        # Two stage (Coarse) Setting, divide training into two phases. Format Reward in [0,0.5] if step < 30 else [0,1]
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step >= 30:
                max_possible_reward = self.format_max_possible / 2
                min_possible_reward = self.format_min_possible / 2
            else:
                max_possible_reward = self.format_max_possible
                min_possible_reward = self.format_min_possible

        # apply continuous interpolation between the two reward scales throughout training.
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = 2 - (2 - max_possible_reward) * global_step / 150
            min_possible_reward = -2 + (2 + min_possible_reward) * global_step / 150
            if max_possible_reward < 1.0:
                max_possible_reward = 1.0
            if min_possible_reward > -1.0:
                min_possible_reward = -1.0

        rewards = []
        responses = completions

        for response, ans in zip(responses, solution):
            reward = min_possible_reward
            if '<response>' in ans and '<tool_call>' not in ans:
                pattern = r'^<think>.*?</think>\s*<response>.*?</response>$'
                if re.search(pattern, response,
                             re.DOTALL) and response.count('<response>') == 1 and response.count('</response>') == 1:
                    reward = max_possible_reward
            elif '<response>' not in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>$'
                if re.search(pattern, response,
                             re.DOTALL) and response.count('<tool_call>') == 1 and response.count('</tool_call>') == 1:
                    reward = max_possible_reward
            elif '<response>' in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>\s*<response>.*?</response>$'
                if (re.search(pattern, response, re.DOTALL) and response.count('<tool_call>') == 1
                        and response.count('</tool_call>') == 1 and response.count('<response>') == 1
                        and response.count('</response>') == 1):
                    reward = max_possible_reward
            else:
                pattern = r'^<think>.*?</think>$'
                if re.search(pattern, response, re.DOTALL):
                    reward = max_possible_reward

            rewards.append(reward)

        return rewards


class ToolUseLengthReward(ORM):

    def __init__(self):
        self.length_max_possible = 1.0
        self.length_min_possible = 0.0

    # customized reward functions: length
    def __call__(self, completions, solution, **kwargs):
        max_possible_reward = self.length_max_possible
        min_possible_reward = self.length_min_possible
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        # SCHEDULELENGTH: enable Dynamic Length Reward
        if os.getenv('SCHEDULELENGTH', 0) == '1':
            max_reward_len = (640 - 384) * global_step / 105 + 384
        else:
            max_reward_len = 512
        """Reward function that gives higher scores to longer completions."""
        responses = completions
        rewards = []

        for response, ans in zip(responses, solution):
            if '<think>' not in response or '</think>' not in response:
                rewards.append(min_possible_reward)
                continue
            think_responses = response.split('<think>')[-1].split('</think>')[0].strip()
            reward = round(len(think_responses.split()) / max_reward_len, 2)
            if reward > 1.0:
                reward = 1.0

            final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
            rewards.append(final_reward)

        return rewards


class ToolUseCorrectnessReward(ORM):

    def __init__(self):
        if str(os.getenv('CORRECTMAX1', 0)) == '1':
            self.tool_max_possible = 1.0
            self.tool_min_possible = -1.0
        else:
            self.tool_max_possible = 3.0
            self.tool_min_possible = -3.0

    def match_score(self, list1, list2):
        if list1 == list2:
            return 1.0

        if os.getenv('REFINEDREWARD', 0) == '1':
            if list1 != list2:
                return 0.0

        if not list1 or not list2:
            return 0.0

        count1 = Counter(list1)  # Frequency count for list1
        count2 = Counter(list2)  # Frequency count for list2

        intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
        max_possible = len(list1) + len(list2) - intersection

        return intersection / max_possible if max_possible > 0 else 0.0

    def compute_tool_call_reward(self, gt_tools, pd_tools, max_possible_reward, min_possible_reward):
        if gt_tools == pd_tools:
            return max_possible_reward

        if os.getenv('COARSEREWARD', 0) == '1':
            if gt_tools != pd_tools:
                return min_possible_reward

        gt_names = [tool['name'] for tool in gt_tools]
        pd_names = [tool['name'] for tool in pd_tools]
        score = self.match_score(list(gt_names), list(pd_names))

        local_max_possible = 1.0
        used_pd_indices = set()  # Keep track of matched pd_tools

        for gt_tool in gt_tools:
            gt_name = gt_tool['name']
            gt_params = gt_tool['parameters']

            if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                local_max_possible += 1.0
            else:
                local_max_possible += 1.0 + len(gt_params)

            best_match = None
            best_match_score = 0.0
            best_match_index = -1

            # Find the best matching unused pd_tool
            for i, pd_tool in enumerate(pd_tools):
                if i in used_pd_indices or pd_tool['name'] != gt_name:
                    continue

                if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                    if gt_tool == pd_tool:
                        best_match = pd_tool
                        best_match_index = i
                        best_match_score = 1.0
                        break
                    else:
                        continue

                pd_params = pd_tool['parameters']
                param_score = self.match_score(list(gt_params.keys()), list(pd_params.keys()))

                # Calculate correctness score for parameter values
                correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

                total_score = param_score + correctness_score

                if total_score > best_match_score:
                    best_match_score = total_score
                    best_match = pd_tool
                    best_match_index = i

            if best_match:
                used_pd_indices.add(best_match_index)
                score += best_match_score

        return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward

    # custoimzed reward functions: tool call correctness
    def __call__(self, completions, solution, **kwargs):
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        max_possible_reward = self.tool_max_possible
        min_possible_reward = self.tool_min_possible
        # two stage (Coarse) Setting, divide training into two phases.
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step < 30:
                max_possible_reward = max_possible_reward / 3
                min_possible_reward = min_possible_reward / 3
            else:
                max_possible_reward = max_possible_reward
                min_possible_reward = min_possible_reward
        # apply continuous interpolation between the two reward scales throughout training.
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = (max_possible_reward - 2) * global_step / 150 + 2
            min_possible_reward = (min_possible_reward + 2) * global_step / 150 - 2
            if max_possible_reward > 3.0:
                max_possible_reward = 3.0
            if min_possible_reward < -3.0:
                min_possible_reward = -3.0

        responses = completions
        rewards = []

        for response, ans in zip(responses, solution):
            reward = 0.0

            if '<tool_call>' not in ans:
                # if "<tool_call>" not in response and "</tool_call>" not in response:
                #     reward = max_possible_reward
                # else:
                #     reward = min_possible_reward
                rewards.append(reward)
                continue

            gt_tool_call = ans.split('<tool_call>')[1].split('</tool_call>')[0].strip()
            gt_tools = gt_tool_call.split('\n')
            gt_tools = [json.loads(tool) for tool in gt_tools]  # each diction contains "name" and "parameter"

            try:
                # if the format is not correct, directly give the lowest possible score
                assert '<tool_call>' in response
                assert '</tool_call>' in response
                pd_tools = response.split('<tool_call>')[1].split('</tool_call>')[0].strip().split('\n')
                pd_tools = [json.loads(tool) for tool in pd_tools]
                reward = self.compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward,
                                                       min_possible_reward)  # top reward is 2
            except (ValueError, IndexError, AssertionError):
                reward = min_possible_reward

            rewards.append(reward)

        return rewards


orms['external_math_acc'] = MathAccuracy
orms['external_math_format'] = MathFormat
orms['external_countdown'] = CountdownORM
orms['external_r1v_acc'] = MultiModalAccuracyORM
orms['external_code_reward'] = CodeReward
orms['external_r1v_acc_classification'] = MultiModalAccuracyClassificationORM
orms['external_video_category_iou'] = VideoCategoryIoUReward
orms['external_video_category_iou_v2'] = VideoCategoryIoUReward_v2
orms['external_video_category_iou_v3'] = VideoCategoryIoUReward_v3
orms['external_code_format'] = CodeFormat
orms['external_code_reward_by_judge0'] = CodeRewardByJudge0
orms['external_tooluse_format_reward'] = ToolUseFormatReward
orms['external_tooluse_length_reward'] = ToolUseLengthReward
orms['external_tooluse_correct_reward'] = ToolUseCorrectnessReward
"""
TO CUSTOMIZE REWARD MODEL:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the messages generated by the model during interactions
        and dataset columns as inputs parameters.

    Step 2: Add your reward model plugin to the rm_plugins registry:
        rm_plugins['my_rm_plugin'] = MyRMPlugin

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_model_plugin my_rm_plugin

For GenRM you can refer to swift/llm/plugin/rm_plugin/GenRMPlugin
"""


class CustomizedRMPlugin:
    """
    Customized Reward Model Plugin, same to DefaultRMPlugin

    It assumes that `self.model` is a classification model with a value head(output dimmension 1).
    The first logits value from the model's output is used as the reward score.
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    def __call__(self, inputs):
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)
        reward_inputs.pop('labels')

        with torch.inference_mode():
            return self.model(**reward_inputs).logits[:, 0]


class QwenLongPlugin(DefaultRMPlugin):
    # https://arxiv.org/abs/2505.17667
    # NOTE: you should customize the verified reward function, you can refer to
    # https://github.com/Tongyi-Zhiwen/QwenLong-L1/tree/main/verl/verl/utils/reward_score
    # hf_dataset: https://huggingface.co/datasets/Tongyi-Zhiwen/DocQA-RL-1.6K/viewer/default/train
    # ms_dataset: https://modelscope.cn/datasets/iic/DocQA-RL-1.6K
    def __init__(self, model, template, accuracy_orm=None):
        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig(temperature=0)  # customise your request config here
        self.system = textwrap.dedent("""
            You are an expert in verifying if two answers are the same.

            Your input consists of a problem and two answers: Answer 1 and Answer 2.
            You need to check if they are equivalent.

            Your task is to determine if the two answers are equivalent, without attempting to solve the original problem.
            Compare the answers to verify they represent identical values or meanings,
            even when expressed in different forms or notations.

            Your output must follow this format:
            1) Provide an explanation for why the answers are equivalent or not.
            2) Then provide your final answer in the form of: [[YES]] or [[NO]]

            Problem: {problem_placeholder}
            Answer 1: {answer1_placeholder}
            Answer 2: {answer2_placeholder}
        """)  # noqa
        self.accuracy_orm = accuracy_orm

    def __call__(self, inputs):
        completions = [example['messages'][-1]['content'] for example in inputs]
        ground_truths = [example['reward_model']['ground_truth'] for example in inputs]
        rm_inputs = self.prepare_rm_inputs(inputs, completions, ground_truths)

        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        llm_rewards = self.compute_rewards(results)

        if self.accuracy_orm:
            verified_rewards = self.accuracy_orm(completions, ground_truths)
        else:
            verified_rewards = [0.0] * len(llm_rewards)

        rewards = [max(r1, r2) for r1, r2 in zip(llm_rewards, verified_rewards)]
        return torch.tensor(rewards, dtype=torch.float32)

    def prepare_rm_inputs(self, inputs: List[Dict], completions, ground_truths) -> List[Dict]:
        rm_inputs = []
        for infer_request, completion, ground_truth in zip(inputs, completions, ground_truths):
            # Deep copy to prevent modification of original input
            rm_infer_request = deepcopy(infer_request)
            problem = infer_request['messages'][0]['content']
            start_index = problem.index('</text>')
            end_index = problem.index('Format your response as follows:')
            question = problem[start_index:end_index].replace('</text>', '').strip()
            prompt = self.system.format(
                problem_placeholder=question, answer1_placeholder=completion, answer2_placeholder=ground_truth)

            # Construct new messages tailored for the reward model
            rm_messages = [{'role': 'user', 'content': prompt}]

            # Update the messages in the reward infer request
            rm_infer_request['messages'] = rm_messages
            rm_inputs.append(rm_infer_request)
        return rm_inputs

    @staticmethod
    def extract_reward(model_output: str) -> float:
        match = re.search(r'\[([A-Z]+)\]', model_output)
        if match:
            answer = match.group(1)
            if answer == 'YES':
                return 1.0
            elif answer == 'NO':
                return 0.0
            else:
                logger.warning("Unexpected answer, expected 'YES' or 'NO'.")
                return 0.0
        else:
            logger.warning("Unable to extract reward score from the model's output, setting reward to 0")
            return 0.0  # Or raise ValueError("Format incorrect")

    def compute_rewards(self, results: List[ChatCompletionResponse]) -> List[float]:
        """
        Compute average reward scores from the reward model's outputs.

        Args:
            results (List[ChatCompletionResponse]): A list of results from the reward model.

        Returns:
            List[float]: A list of average reward scores.
        """
        rewards = []
        for idx, output in enumerate(results):
            try:
                cur_rewards = []
                for choice in output.choices:
                    response = choice.message.content
                    reward = self.extract_reward(response)
                    cur_rewards.append(reward)
                cur_rewards = [r for r in cur_rewards if r is not None]
                if cur_rewards:
                    average_reward = sum(cur_rewards) / len(cur_rewards)
                else:
                    average_reward = 0.0
                    logger.warning('No valid rewards extracted. Assigning reward score of 0.0.')

                rewards.append(average_reward)
            except Exception as e:
                logger.error(f'Error computing reward: {e}')
                rewards.append(0.0)  # Assign default reward score on failure
        return rewards


rm_plugins['my_rmplugin'] = CustomizedRMPlugin
rm_plugins['qwenlong'] = QwenLongPlugin
"""
TO CUSTOMIZE MULTITURN SCHEDULER:
    Step 1: Define a Scheduler Class
        Implement your custom scheduler with the following methods:
            - step() (Required): Constructs the next round of the infer request.
            - check_finished() (Optional): Determines whether the current round has finished,
                which defaults to ending when the inference result is truncated (over length) or
                when the maximum number of rounds is reached.
        Both methods accept
            - the last turn's InferRequest/result
            The current turn count

    Step 2: Add your scheduler to the multi_turns registry:
        multi_turns['my_scheduler'] = MyScheduler

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --multi_turn_scheduler my_scheduler
"""


class ReToolScheduler(MultiTurnScheduler):
    pass


multi_turns['retool'] = ReToolScheduler


# register GYM env
class CustomEnv(Env):
    pass


envs['custom_env'] = CustomEnv


class CustomCtxManager(ContextManager):
    pass


context_managers['custom_ctx'] = CustomCtxManager


class DrivingVideoClassificationReward(ORM):
    """
    汽车驾驶视频多分类任务的奖励函数
    包含增强的准确率奖励和推理长度奖励
    """
    
    def __init__(self, accuracy_weight=0.8, length_weight=0.2, accuracy_enhancement_power=2.0):
        """
        初始化奖励函数
        
        Args:
            accuracy_weight: 准确率奖励权重
            length_weight: 推理长度奖励权重  
            accuracy_enhancement_power: 准确率奖励增强幂次，用于增强准确率信号
        """
        self.accuracy_weight = accuracy_weight
        self.length_weight = length_weight
        self.accuracy_enhancement_power = accuracy_enhancement_power
        
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        计算奖励分数
        
        Args:
            completions: 模型输出列表
            solution: 标准答案列表
            **kwargs: 其他参数
            
        Returns:
            list[float]: 每个样本的奖励分数
        """
        logger.info(f"DrivingVideoClassificationReward completions: {completions}")
        logger.info(f"DrivingVideoClassificationReward solution: {solution}")
        
        rewards = []
        compared_labels_list = []
        for completion, gt in zip(completions, solution):
            # 1. 提取答案部分
            gt_match = re.search(r'<answer>(.*?)</answer>', gt)
            ground_truth = gt_match.group(1).strip() if gt_match else gt.strip()

            completion_match = re.search(r'<answer>(.*?)</answer>', completion)
            if not completion_match:
                rewards.append(0.0)
                continue
            predicted_answer = completion_match.group(1).strip()
            
            # 2. 解析多分类标签
            predicted_labels = self._parse_labels(predicted_answer)
            ground_truth_labels = self._parse_labels(ground_truth)

            compared_labels = {
                "predicted_labels": list(predicted_labels),
                "ground_truth_labels": list(ground_truth_labels)
            }
            compared_labels_list.append(compared_labels)
            
            # 3. 计算多分类准确率（F1分数）并增强准确率信号
            accuracy_reward = self._calculate_multiclass_accuracy(predicted_labels, ground_truth_labels)
            if accuracy_reward == 0.0:
                rewards.append(0.0)
                continue
            # 使用幂次变换增强准确率奖励的方差
            # 当 accuracy_enhancement_power=2.0 时：
            # 当 accuracy_reward=0.8 时，enhanced_accuracy_reward=0.8^0.5=0.894
            # 当 accuracy_reward=0.9 时，enhanced_accuracy_reward=0.9^0.5=0.949
            # 这样可以让高准确率获得更高的奖励，低准确率获得更低的奖励
            enhanced_accuracy_reward = accuracy_reward ** (1.0 / self.accuracy_enhancement_power)
            
            # 4. 计算推理长度奖励
            think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
            if think_match:
                reasoning_text = think_match.group(1).strip()
                # 计算推理文本长度（字符数）
                reasoning_length = len(reasoning_text)
                # 归一化长度奖励（假设理想长度为200-500字符）
                if reasoning_length < 50:
                    length_reward = 0.0
                elif reasoning_length < 200:
                    length_reward = reasoning_length / 200.0
                elif reasoning_length <= 500:
                    length_reward = 1.0
                else:
                    # 超过500字符时逐渐减少奖励
                    length_reward = max(0.0, 1.0 - (reasoning_length - 500) / 300.0)
            else:
                length_reward = 0.0
            
            # 5. 组合最终奖励（移除方差惩罚，增强准确率信号）
            final_reward = (
                self.accuracy_weight * enhanced_accuracy_reward +
                self.length_weight * length_reward
            )
            
            # 确保奖励在[0, 1]范围内
            final_reward = max(0.0, min(1.0, final_reward))
            
            rewards.append(final_reward)

        logger.info(f"DrivingVideoClassificationReward labels compare: {compared_labels_list}")
        logger.info(f"DrivingVideoClassificationReward rewards: {rewards}")
            
        return rewards
    
    def _parse_labels(self, label_str: str) -> set:
        """
        解析标签字符串为标签集合
        
        Args:
            label_str: 标签字符串，可以是逗号分隔的多个标签
            
        Returns:
            set: 标签集合
        """
        if not label_str:
            return set()
        
        # 分割标签并清理
        labels = [label.strip() for label in label_str.split(',')]
        # 过滤空标签
        labels = [label for label in labels if label]
        return set(labels)
    
    def _calculate_multiclass_accuracy(self, predicted_labels: set, ground_truth_labels: set) -> float:
        """
        计算多分类准确率（使用F1分数）
        
        Args:
            predicted_labels: 预测标签集合
            ground_truth_labels: 真实标签集合
            
        Returns:
            float: F1分数 (0-1)
        """
        if not ground_truth_labels:
            return 1.0 if not predicted_labels else 0.0
        
        if not predicted_labels:
            return 0.0
        
        # 计算精确率和召回率
        intersection = predicted_labels & ground_truth_labels
        precision = len(intersection) / len(predicted_labels) if predicted_labels else 0.0
        recall = len(intersection) / len(ground_truth_labels) if ground_truth_labels else 0.0
        
        # 计算F1分数
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return f1


class DrivingVideoClassificationRewardV2(ORM):
    """
    改进版汽车驾驶视频多分类奖励函数
    增加更细致的奖励计算和错误分析，并增强准确率信号
    """
    
    def __init__(self, accuracy_weight=0.7, length_weight=0.15, format_weight=0.15, accuracy_enhancement_power=2.0):
        """
        初始化奖励函数
        
        Args:
            accuracy_weight: 准确率奖励权重
            length_weight: 推理长度奖励权重
            format_weight: 格式正确性奖励权重
            accuracy_enhancement_power: 准确率奖励增强幂次，用于增强准确率信号
        """
        self.accuracy_weight = accuracy_weight
        self.length_weight = length_weight
        self.format_weight = format_weight
        self.accuracy_enhancement_power = accuracy_enhancement_power
        
        # 定义有效的分类标签
        self.valid_categories = {
            '白天驾驶', 'Day Driving', 'day driving',
            '夜间驾驶', 'Night Driving', 'night driving', 
            '城市道路', 'Urban Road', 'urban road',
            '高速公路', 'Highway', 'highway',
            '乡村道路', 'Rural Road', 'rural road',
            '雨天驾驶', 'Rainy Driving', 'rainy driving',
            '雪天驾驶', 'Snowy Driving', 'snowy driving',
            '拥堵交通', 'Traffic Jam', 'traffic jam',
            '停车场景', 'Parking', 'parking',
            '其他', 'Other', 'other'
        }
        
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        计算奖励分数
        
        Args:
            completions: 模型输出列表
            solution: 标准答案列表
            **kwargs: 其他参数
            
        Returns:
            list[float]: 每个样本的奖励分数
        """
        logger.info(f"DrivingVideoClassificationRewardV2 completions: {completions}")
        logger.info(f"DrivingVideoClassificationRewardV2 solution: {solution}")
        
        rewards = []
        for completion, gt in zip(completions, solution):
            # 1. 格式检查
            format_score = self._check_format(completion)
            
            # 2. 提取答案
            answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
            if not answer_match:
                rewards.append(0.0)
                continue
                
            predicted_answer = answer_match.group(1).strip()
            ground_truth = gt.strip()
            
            # 3. 解析多分类标签
            predicted_labels = self._parse_labels(predicted_answer)
            ground_truth_labels = self._parse_labels(ground_truth)
            
            # 4. 多分类准确率计算并增强准确率信号
            accuracy_score = self._calculate_multiclass_accuracy(predicted_labels, ground_truth_labels)
            # 使用幂次变换增强准确率奖励的方差
            # 当 accuracy_enhancement_power=2.0 时：
            # 当 accuracy_score=0.8 时，enhanced_accuracy_score=0.8^0.5=0.894
            # 当 accuracy_score=0.9 时，enhanced_accuracy_score=0.9^0.5=0.949
            # 这样可以让高准确率获得更高的奖励，低准确率获得更低的奖励
            enhanced_accuracy_score = accuracy_score ** (1.0 / self.accuracy_enhancement_power)
            
            # 5. 推理长度计算
            length_score = self._calculate_length_score(completion)
            
            # 6. 组合最终奖励
            final_reward = (
                self.accuracy_weight * enhanced_accuracy_score +
                self.length_weight * length_score +
                self.format_weight * format_score
            )
            
            # 确保奖励在[0, 1]范围内
            final_reward = max(0.0, min(1.0, final_reward))
            
            rewards.append(final_reward)
            
        return rewards
    
    def _parse_labels(self, label_str: str) -> set:
        """
        解析标签字符串为标签集合
        
        Args:
            label_str: 标签字符串，可以是逗号分隔的多个标签
            
        Returns:
            set: 标签集合
        """
        if not label_str:
            return set()
        
        # 分割标签并清理
        labels = [label.strip() for label in label_str.split(',')]
        # 过滤空标签
        labels = [label for label in labels if label]
        return set(labels)
    
    def _calculate_multiclass_accuracy(self, predicted_labels: set, ground_truth_labels: set) -> float:
        """
        计算多分类准确率（使用F1分数）
        
        Args:
            predicted_labels: 预测标签集合
            ground_truth_labels: 真实标签集合
            
        Returns:
            float: F1分数 (0-1)
        """
        if not ground_truth_labels:
            return 1.0 if not predicted_labels else 0.0
        
        if not predicted_labels:
            return 0.0
        
        # 计算精确率和召回率
        intersection = predicted_labels & ground_truth_labels
        precision = len(intersection) / len(predicted_labels) if predicted_labels else 0.0
        recall = len(intersection) / len(ground_truth_labels) if ground_truth_labels else 0.0
        
        # 计算F1分数
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    def _check_format(self, completion: str) -> float:
        """
        检查输出格式是否正确
        
        Args:
            completion: 模型输出
            
        Returns:
            float: 格式分数 (0-1)
        """
        # 检查是否包含必要的标签
        has_think = '<think>' in completion and '</think>' in completion
        has_answer = '<answer>' in completion and '</answer>' in completion
        
        if not (has_think and has_answer):
            return 0.0
            
        # 检查标签顺序
        think_start = completion.find('<think>')
        think_end = completion.find('</think>')
        answer_start = completion.find('<answer>')
        answer_end = completion.find('</answer>')
        
        if not (think_start < think_end < answer_start < answer_end):
            return 0.0
            
        # 检查内容是否为空
        think_content = completion[think_start + 7:think_end].strip()
        answer_content = completion[answer_start + 8:answer_end].strip()
        
        if not think_content or not answer_content:
            return 0.5
            
        return 1.0
    
    def _calculate_length_score(self, completion: str) -> float:
        """
        计算推理长度分数
        
        Args:
            completion: 模型输出
            
        Returns:
            float: 长度分数 (0-1)
        """
        think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
        if not think_match:
            return 0.0
            
        reasoning_text = think_match.group(1).strip()
        reasoning_length = len(reasoning_text)
        
        # 理想长度范围：100-400字符
        if reasoning_length < 50:
            return 0.0
        elif reasoning_length < 100:
            return reasoning_length / 100.0
        elif reasoning_length <= 400:
            return 1.0
        else:
            # 超过400字符时逐渐减少奖励
            return max(0.0, 1.0 - (reasoning_length - 400) / 200.0)


class DrivingVideoClassificationNoThinkReward(ORM):
    """
    汽车驾驶视频多分类任务的奖励函数
    包含增强的准确率奖励和推理长度奖励
    """

    def __init__(self, accuracy_weight=0.8, length_weight=0.2, accuracy_enhancement_power=2.0):
        """
        初始化奖励函数

        Args:
            accuracy_weight: 准确率奖励权重
            length_weight: 推理长度奖励权重
            accuracy_enhancement_power: 准确率奖励增强幂次，用于增强准确率信号
        """
        self.accuracy_weight = accuracy_weight
        self.length_weight = length_weight
        self.accuracy_enhancement_power = accuracy_enhancement_power

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        计算奖励分数

        Args:
            completions: 模型输出列表
            solution: 标准答案列表
            **kwargs: 其他参数

        Returns:
            list[float]: 每个样本的奖励分数
        """
        logger.info(f"DrivingVideoClassificationNoThinkReward completions: {completions}")
        logger.info(f"DrivingVideoClassificationNoThinkReward solution: {solution}")

        rewards = []
        compared_labels_list = []
        for completion, gt in zip(completions, solution):
            # 1. 提取答案部分
            gt_match = re.search(r'<answer>(.*?)</answer>', gt)
            ground_truth = gt_match.group(1).strip() if gt_match else gt.strip()

            completion_match = re.search(r'<answer>(.*?)</answer>', completion)
            if not completion_match:
                rewards.append(0.0)
                continue
            predicted_answer = completion_match.group(1).strip()

            # 2. 解析多分类标签
            predicted_labels = self._parse_labels(predicted_answer)
            ground_truth_labels = self._parse_labels(ground_truth)

            compared_labels = {
                "predicted_labels": list(predicted_labels),
                "ground_truth_labels": list(ground_truth_labels)
            }
            compared_labels_list.append(compared_labels)

            # 3. 计算多分类准确率（F1分数）并增强准确率信号
            accuracy_reward = self._calculate_multiclass_accuracy(predicted_labels, ground_truth_labels)
            accuracy_reward = accuracy_reward * len(ground_truth_labels)
            if accuracy_reward == 0.0:
                rewards.append(0.0)
                continue
            # 使用幂次变换增强准确率奖励的方差
            # 当 accuracy_enhancement_power=2.0 时：
            # 当 accuracy_reward=0.8 时，enhanced_accuracy_reward=0.8^0.5=0.894
            # 当 accuracy_reward=0.9 时，enhanced_accuracy_reward=0.9^0.5=0.949
            # 这样可以让高准确率获得更高的奖励，低准确率获得更低的奖励
            final_reward = accuracy_reward ** (1.0 / self.accuracy_enhancement_power)
            
            # 确保奖励在[0, 1]范围内
            # final_reward = max(0.0, min(1.0, final_reward))

            rewards.append(final_reward)

        logger.info(f"DrivingVideoClassificationNoThinkReward labels compare: {compared_labels_list}")
        logger.info(f"DrivingVideoClassificationNoThinkReward rewards: {rewards}")

        return rewards

    def _parse_labels(self, label_str: str) -> set:
        """
        解析标签字符串为标签集合

        Args:
            label_str: 标签字符串，可以是逗号分隔的多个标签

        Returns:
            set: 标签集合
        """
        if not label_str:
            return set()

        # 分割标签并清理
        labels = [label.strip() for label in label_str.split(',')]
        # 过滤空标签
        labels = [label for label in labels if label]
        return set(labels)

    def _calculate_multiclass_accuracy(self, predicted_labels: set, ground_truth_labels: set) -> float:
        """
        计算多分类准确率（使用F1分数）

        Args:
            predicted_labels: 预测标签集合
            ground_truth_labels: 真实标签集合

        Returns:
            float: F1分数 (0-1)
        """
        if not ground_truth_labels:
            return 1.0 if not predicted_labels else 0.0

        if not predicted_labels:
            return 0.0

        # 计算精确率和召回率
        intersection = predicted_labels & ground_truth_labels
        precision = len(intersection) / len(predicted_labels) if predicted_labels else 0.0
        recall = len(intersection) / len(ground_truth_labels) if ground_truth_labels else 0.0

        # 计算F1分数
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return f1


class DrivingVideoMultiClassificationReward(ORM):
    """
    汽车驾驶视频多分类任务的奖励函数，支持多标签分类，每个标签之间用逗号分隔
    """

    def __init__(self):
        pass

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        计算奖励分数

        Args:
            completions: 模型输出列表
            solution: 标准答案列表
            **kwargs: 其他参数

        Returns:
            list[float]: 每个样本的奖励分数
        """

        rewards = []
        compared_labels_list = []
        for completion, gt in zip(completions, solution):
            # 1. 提取答案部分
            gt_match = re.search(r'<answer>(.*?)</answer>', gt)
            ground_truth = gt_match.group(1).strip() if gt_match else gt.strip()

            completion_match = re.search(r'<answer>(.*?)</answer>', completion)
            if not completion_match:
                rewards.append(0.0)
                continue
            predicted_answer = completion_match.group(1).strip()

            # 2. 解析多分类标签
            predicted_labels = self._parse_labels(predicted_answer)
            ground_truth_labels = self._parse_labels(ground_truth)

            compared_labels = {
                "predicted_labels": list(predicted_labels),
                "ground_truth_labels": list(ground_truth_labels)
            }
            compared_labels_list.append(compared_labels)

            # Check if student answer is a subset of correct answers
            if predicted_labels.issubset(ground_truth_labels):
                # Calculate partial credit: number of correct answers / total number of correct answers
                reward = len(predicted_labels) / len(ground_truth_labels)
            else:
                reward = 0.0


            rewards.append(reward)

        logger.info(f"completions: {completions}")
        logger.info(f"solution: {solution}")
        video_path = kwargs.get("videos", None)
        if not video_path:
            logger.warning(f"kwargs: {kwargs}")

        logger.info(f"video_path: {video_path}")
        logger.info(f"labels compare: {compared_labels_list}")
        logger.info(f"rewards: {rewards}")

        return rewards

    def _parse_labels(self, label_str: str) -> set:
        """
        解析标签字符串为标签集合

        Args:
            label_str: 标签字符串，可以是逗号分隔的多个标签

        Returns:
            set: 标签集合
        """
        if not label_str:
            return set()

        # 分割标签并清理
        labels = [label.strip() for label in label_str.split(',')]
        # 过滤空标签
        labels = [label for label in labels if label]
        return set(labels)


# 注册新的奖励函数
orms['driving_video_classification_reward'] = DrivingVideoClassificationReward
orms['driving_video_classification_reward_v2'] = DrivingVideoClassificationRewardV2
orms['driving_video_classification_reward_no_think'] = DrivingVideoClassificationNoThinkReward
orms['driving_video_multi_classification_reward'] = DrivingVideoMultiClassificationReward
