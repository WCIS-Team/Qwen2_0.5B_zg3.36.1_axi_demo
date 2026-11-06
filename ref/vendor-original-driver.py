import numpy as np
from tokenizers import Tokenizer
import os 
import sys 
from typing import List,Dict,Optional,Tuple
import datetime
import time
#添加icraft 
import icraft 
from icraft import xir
#添加自定义模块
sys.path.append(R"./modelzoo_utils")
from pyrtutils.icraft_utils import *  
from pyrtutils.utils import *
from pyrtutils.et_device import * 
from pyrtutils.calctime_utils import *


class ModelConfig:
    "模型配置参数"
    config_path: str = R"config/Qwen2-0___5B-Instruct"
    decode_json: str = R"jr_path/decode/Qwen2-0.5B-decoder_ZG.json"
    decode_raw: str = R"jr_path/decode/Qwen2-0.5B-decoder_ZG.raw" 
    
    max_decode_length: int = 512
    max_new_tokens: int = 150  #生成token达到该长度后，停止生成
    dtype_kv: np.dtype = np.float16
    run_backend: str = "zg330"
    ip: str = "192.168.125.143"
    mmu_mode: bool = True
    cuda_mode: bool = False 
    vocab_size: int = 151936
    # 默认采样配置参数
    sampling_method: str = "greedy"  # 支持 "greedy", "top_k", "top_p"
    sampling_value: float = 50     # top_k或top_p的值
    temperature: float = 1.0         # 温度参数
    repetition_penalty: float = 1.2  # 重复惩罚参数


class ContentFilter:
    """智能内容过滤器，使用通用方法检测生成质量"""
    
    def __init__(self):
        # 通用检测参数
        self.max_repetition_ratio = 0.6  # 最大重复率阈值
        self.min_meaningful_tokens = 10  # 最少有意义的token数
        self.max_token_repetition = 0.8  # token重复率阈值
        
    def should_stop_generation(self, generated_tokens: List[int], tokenizer, step: int) -> bool:
        """判断是否应该停止生成，基于通用质量检测"""
        if len(generated_tokens) < self.min_meaningful_tokens:
            return False
        
        # 1. 检查token重复率
        if self._check_token_repetition(generated_tokens):
            print("High token repetition detected, stopping generation")
            return True
        
        # 2. 检查内容多样性
        if self._check_content_diversity(generated_tokens, tokenizer):
            print("Low content diversity detected, stopping generation")
            return True
        
        # 3. 检查生成是否陷入循环
        if self._check_generation_loop(generated_tokens):
            print("Generation loop detected, stopping generation")
            return True
        
        return False
    
    def _check_token_repetition(self, generated_tokens: List[int]) -> bool:
        """检查token重复率"""
        if len(generated_tokens) < 20:
            return False
        
        # 检查最近20个token的重复率
        recent_tokens = generated_tokens[-20:]
        unique_ratio = len(set(recent_tokens)) / len(recent_tokens)
        
        return unique_ratio < 0.3  # 如果重复率超过70%
    
    def _check_content_diversity(self, generated_tokens: List[int], tokenizer) -> bool:
        """检查内容多样性"""
        if len(generated_tokens) < 30:
            return False
        
        # 解码文本并分析
        text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        
        # 简单的句子结构分析（不使用re模块）
        sentences = []
        current_sentence = ""
        for char in text:
            current_sentence += char
            if char in '.!?。！？':
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if len(sentences) < 2:
            return False
        
        # 检查句子多样性（简单的词汇多样性检查）
        words = []
        current_word = ""
        for char in text:
            if char.isalnum() or char in '_-':
                current_word += char
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ""
        
        if len(words) < 20:
            return False
        
        word_diversity = len(set(words)) / len(words)
        return word_diversity < 0.4  # 词汇多样性低于40%
    
    def _check_generation_loop(self, generated_tokens: List[int]) -> bool:
        """检查是否陷入生成循环"""
        if len(generated_tokens) < 40:
            return False
        
        # 检查是否存在明显的循环模式
        # 简单的检查：最近20个token是否与之前的20个token高度相似
        if len(generated_tokens) >= 40:
            recent_20 = generated_tokens[-20:]
            prev_20 = generated_tokens[-40:-20]
            
            # 简单的相似度检查
            if recent_20 == prev_20:
                return True
            
            # 检查部分重叠
            overlap = len(set(recent_20) & set(prev_20)) / 20
            if overlap > 0.8:  # 80%的重叠
                return True
        
        return False

def Tensor2Numpy(outputs: List[icraft.xrt.Tensor]) -> List[np.ndarray]:
    """
    Convert icraft.Tensor to numpy.ndarray.
    """
    # Normalize inputs to list
    if not isinstance(outputs, list):
        outputs = [outputs]
    
    numpy_arrays = []
    for idx, item in enumerate(outputs):
        try:
            chunk = item.chunk()
            byte_size = chunk.byte_size
            offset = item.offset()
            tensor_shape = item.dtype().shape
            input_dtype = item.dtype().element_dtype
            
            if str(input_dtype) == '"@fp(32)"':
                np_dtype = np.float32
            elif str(input_dtype) == '"@fp(16)"':
                np_dtype = np.float16
            elif str(input_dtype) == '"@uint(8)"':
                np_dtype = np.uint8
            elif str(input_dtype) == '"@uint(16)"':
                np_dtype = np.uint16
            elif str(input_dtype) == '"@sint(8)"':
                np_dtype = np.int8
            elif str(input_dtype) == '"@sint(16)"':
                np_dtype = np.int16
            # Create numpy buffer and read tensor data
            py_buffer = np.zeros(shape=tensor_shape, dtype=np_dtype)
            #到时候需要进一步判断 需要转换的数据类型
            item.read(py_buffer, offset, byte_size)
            numpy_arrays.append(py_buffer)
        except Exception as e:
            raise RuntimeError(f"Failed to convert tensor at index {idx}: {e}")
    
    return numpy_arrays

class IcraftModel:
    def __init__(self,config:ModelConfig):
        self.config = config 
        self.tokenizer = None 
        self.model_cfg = None 
        self.device = None 
    def load_model(self):
        "加载模型和tokenizer"
        # 使用tokenizers库的tokenizer
        tokenizer_path = os.path.join(self.config.config_path, "tokenizer.json") if self.config.config_path else None
        self.tokenizer = Tokenizer.from_file(tokenizer_path) 

        return self 
    def init_device(self):
        "初始化设备"
        if self.device is not None:
            return self 
        self.device = openDevice(
            self.config.run_backend,
            self.config.ip,
            self.config.mmu_mode,
            self.config.cuda_mode
        )
        return self 
    def _load_network(self,json_path:str,raw_path:str) -> xir.Network:
        "create Netowork"
        network = icraft.xir.Network.CreateFromJsonFile(json_path)
        network.lazyLoadParamsFromFile(raw_path)
        return network
    
    def reset_device(self):
        "重置设备"
        if self.config.run_backend != "host" and self.device:
            self.device.reset(1)


class DecodeOnlySessionManager:
    "只使用decode网络的session管理类"
    def __init__(self,model:IcraftModel):
        self.model = model 
        self.model.load_model().init_device()

        self.n_layers =24
        self.n_kv_heads = 2
        self.head_dim = 64

        # 只使用decode网络
        self.decode_network = None
        self.decode_session = None

        self.kv_dtype = None
        # self.kv_elementdtype = None
        
        self._init_decode_network()
    
    def _init_decode_network(self):
        """只初始化decode网络"""
        # 加载decode网络
        self.decode_network = self.model._load_network(
            self.model.config.decode_json,
            self.model.config.decode_raw,
        )

        # 创建decode session
        self.decode_session = initSession(
            self.model.config.run_backend,
            self.decode_network,
            self.model.device,
            1
        )

        # self.decode_session.enableTimeProfile(True)

        self.decode_session.apply()

        self.kv_dtype = self.decode_network.ops[0].outputs[2].tensorType()
        # self.kv_elementdtype = self.kv_dtype.element_dtype
        
        print("Decode-only network and session initialized successfully")
    
    def _pad_to_len(self, arr: np.ndarray, pad_len: int, pad_val: int = 0, 
                dtype: np.dtype = np.float32) -> np.ndarray:
        """将数组填充到指定长度"""
        if arr.shape[1] >= pad_len:
            return arr[:, :pad_len].astype(dtype)
        
        pad = np.full((arr.shape[0], pad_len - arr.shape[1]), pad_val, dtype=dtype)
        return np.concatenate([arr, pad], axis=1).astype(dtype)




    def _prepare_kv_cache(self,batch_size: int, seq_len: int) -> List[np.ndarray]:
        "准备KV缓存"
        return [
            np.zeros((batch_size, self.n_kv_heads, seq_len, self.head_dim), dtype=self.model.config.dtype_kv)
            for _ in range(self.n_layers * 2)
        ]
    def _create_feed_dict(self, input_ids: np.ndarray, token_dynamic_fm: np.ndarray, kv_cache: List[np.ndarray]) -> Dict[str, np.ndarray]:
        "创建feed dict"
        feed = {
            "input_ids": input_ids,
            "token_dynamic_fm": token_dynamic_fm
        }
        # prepare KV cache 
        for i in range(self.n_layers ):
            feed[f'past_key_values.{i}.key'] = kv_cache[i * 2]
            feed[f'past_key_values.{i}.value'] = kv_cache[i * 2 + 1]
        return feed
    
    def _numpy2Tensor(self,feed:Dict[str,np.ndarray], network: xir.Network) -> List:
        "将numpy数组转换为Icraft Tensor"
        temp_list=[]
        for idx, v in enumerate(feed.values()):
            # tem_tensor = numpy2Tensor(v, network, idx)
            if idx == 0  or idx == 1:
                temp_list.append(Tensor(v.astype(np.float32),xir.Layout("*C")))
            else:
                temp_list.append(Tensor(v.astype(np.float16),self.kv_dtype))
        return temp_list
    # 采样方法
    def sample_logits(self,logits: np.ndarray, generated_tokens: List[int] = None, sampling_method: str = "greedy",sampling_value: float = None,temperature: float = 1.0, repetition_penalty: float = 1.0) -> np.ndarray:
        """对logits进行采样，支持重复惩罚"""
        
        # 应用重复惩罚
        if repetition_penalty != 1.0 and generated_tokens:
            logits = self._apply_repetition_penalty(logits, generated_tokens, repetition_penalty)
        
        if temperature == 0 or sampling_method == "greedy":
            # 直接取logits中最大的索引
            next_token = np.argmax(logits,axis=-1).astype(np.int64)
        elif sampling_method == "top_k" or sampling_method == "top_p":
            assert sampling_value is not None
            logits = logits.astype(np.float32)
            logits /= temperature
            probs = np.exp(logits) / np.sum(np.exp(logits))
            sorted_probs = np.sort(probs)[:, ::-1]
            sorted_indices = np.argsort(probs)[:, ::-1]

            if sampling_method == "top_k":
                index_of_interest = int(sampling_value)
            elif sampling_method == "top_p":
                p = sampling_value
                cumulative_probs = np.cumsum(sorted_probs, axis=-1)
                for index_of_interest, cumulative_prob in enumerate(
                    cumulative_probs[0]
                ):
                    if cumulative_prob > p:
                        break

            probs_of_interest = sorted_probs[:, : index_of_interest + 1]
            indices_of_interest = sorted_indices[:, : index_of_interest + 1]
            probs_of_interest /= np.sum(probs_of_interest)
            next_token = np.array(
                [np.random.choice(indices_of_interest[0], p=probs_of_interest[0])]
            )
        else:
            raise Exception(f"Unknown sampling method {sampling_method}")

        return next_token
    
    def _apply_repetition_penalty(self, logits: np.ndarray, generated_tokens: List[int], penalty: float) -> np.ndarray:
        """应用重复惩罚"""
        if penalty == 1.0 or not generated_tokens:
            return logits
        
        # 对最近生成的token应用惩罚
        recent_tokens = generated_tokens[-10:]  # 只考虑最近10个token
        for token in set(recent_tokens):
            logits[0, token] /= penalty
        
        return logits

    def process_prompt_character_by_character(self, prompt_tokens: List[int]) -> Tuple[int, List[np.ndarray]]:
        """使用decode网络逐字符处理prompt，模拟prefill功能"""
        if self.decode_session is None:
            raise RuntimeError("Decode session not initialized")
        
        # 初始化空的KV缓存
        kv_cache = self._prepare_kv_cache(batch_size=1, seq_len=self.model.config.max_decode_length)
        # print(f"开始逐字符处理prompt，共{len(prompt_tokens)}个token")
        
        # 逐字符处理prompt
        for i, token in enumerate(prompt_tokens):
            # 准备输入
            token_dynamic_fm = np.array([[1, i+1]], dtype=np.float32)  # seq_len=1, pos=i
            input_ids = np.array([[token]], dtype=np.int64)
            
            # 创建feed dict
            feed = self._create_feed_dict(input_ids, token_dynamic_fm, kv_cache)
            input_tensors = self._numpy2Tensor(feed, self.decode_network)

            # 运行decode推理
            outputs = self.decode_session.forward(input_tensors)
            # calctime_detail(self.model.config.run_backend,self.decode_session,self.decode_network,"qwen2_0.5b_socket_decode.xlsx")
            # 手动同步
            for tensor in outputs:
                timeout = datetime.timedelta(milliseconds=50000)
                tensor.waitForReady(timeout)

            self.model.reset_device()
            output_arrays = Tensor2Numpy(outputs)
            # 更新KV缓存
            new_kv_cache = output_arrays[1:]
            
            # 将新的KV缓存更新到大的KV缓存表中
            for j, new_kv in enumerate(new_kv_cache):
                # 更新当前位置的KV缓存
                kv_cache[j][:, :, i, :] = new_kv[:, :, i+1, :]
            # if (i + 1) % 10 == 0:
            #     print(f"已处理 {i+1}/{len(prompt_tokens)} 个token")
        
        print("Prompt处理完成")
        # 获取最后一个token的logits用于生成第一个输出token
        logits = output_arrays[0].reshape(1,1,-1)  # shape: (batch, 1, vocab_size)
        current_token_logits = logits[:, 0, :]  # shape: (batch, vocab_size)

        # 使用sample_logits进行采样
        next_token_array = self.sample_logits(
            current_token_logits,
            [],  # 还没有生成的token
            self.model.config.sampling_method,
            self.model.config.sampling_value,
            self.model.config.temperature,
            self.model.config.repetition_penalty
        )
        first_token = int(next_token_array[0])

        return first_token, kv_cache
    
    def run_decode(self, input_token: int, kv_cache: List[np.ndarray], seq_pos: int, generated_tokens: List[int]) -> Tuple[int, List[np.ndarray]]:
        "运行decode推理"
        # 使用已初始化的decode session
        if self.decode_session is None:
            raise RuntimeError("Decode session not initialized")
        #准备输入 
        decode_seq_len = 1
        token_dynamic_fm = np.array([[decode_seq_len, seq_pos]], dtype=np.float32)
        input_ids = np.array([[input_token]], dtype=np.float32)

        feed = self._create_feed_dict(input_ids, token_dynamic_fm, kv_cache)
        input_tensors = self._numpy2Tensor(feed, self.decode_network)
        # 运行推理
        outputs = self.decode_session.forward(input_tensors)
        # calctime_detail(self.model.config.run_backend,self.decode_session,self.decode_network,"qwen2_0.5b_axi_decode.txt")
        # 手动同步
        for tensor in outputs:
            timeout = datetime.timedelta(milliseconds=50000)
            tensor.waitForReady(timeout)

        self.model.reset_device()
        output_arrays = Tensor2Numpy(outputs)
       
        # 提取第一个token的logits, sample_logits
        logits = output_arrays[0].reshape(1,1,-1)  # shape: (batch, 1, vocab_size)

        # 提取第一个位置的logits
        current_token_logits = logits[:, 0, :]  # shape: (batch, vocab_size)

        # 使用sample_logits进行采样
        next_token_array = self.sample_logits(
            current_token_logits,
            generated_tokens,  # 传递已生成的token用于重复惩罚
            self.model.config.sampling_method,
            self.model.config.sampling_value,
            self.model.config.temperature,
            self.model.config.repetition_penalty
        )
        next_token = int(next_token_array[0])
        
        #更新kv_cache 
        new_kv_cache = output_arrays[1:]

        return next_token, new_kv_cache
    

class QwenGenerator:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = IcraftModel(self.config)
        self.session_mgr = DecodeOnlySessionManager(self.model)
        self.content_filter = ContentFilter()
        
    def generate(self,prompt : str, max_length: Optional[int] = None) -> str:
        "生成文本（使用decode-only模式）"
        max_length = max_length or self.config.max_new_tokens
        # 性能指标计时开始
        total_start_time = time.time()
        
        # Tokenize 输入
        new_prompt = "<|im_start|>\n"+prompt+"<|im_end|>assistant\n"  # 这样最终只会增加4个token
        # inputs = self.model.tokenizer(new_prompt, return_tensors="np")
        # prompt_tokens = inputs["input_ids"][0].tolist()  # 获取token列表
        # 使用轻量级tokenizer的encode方法
        prompt_tokens = self.model.tokenizer.encode(new_prompt).ids #self.model.tokenizer.encode(new_prompt, add_special_tokens=False)
        # tokenizer_time = time.time()
        # print(f"Tokenizer耗时: {(tokenizer_time - total_start_time)*1000:.4f}ms")
        
        prompt_length = len(prompt_tokens)
        # print("====after tokenizer:", prompt_tokens)
        # 使用decode网络逐字符处理prompt
        print('---STAGE: Prefill... Processing prompt with decode network...')
        prefill_start_time = time.time()
        next_token, kv_cache = self.session_mgr.process_prompt_character_by_character(prompt_tokens)
        prefill_end_time = time.time()
        prefill_time_ms = (prefill_end_time - prefill_start_time) * 1000
        
        generated = [next_token]
        print(f"First generated token: {next_token}")

        # decode阶段 
        print('---STAGE: Decode...')
        eos_id = 151643 #self.model.tokenizer.eos_token_id
        #--- 新增：Decode阶段计时列表
        decode_times_ms = []
        
        for step in range(1, max_length):
            if next_token == eos_id:
                print(f"EOS token, stop generating")
                break
            if step == max_length - 1:
                print(f"Max length reached, stop generating, max_length = {max_length}")
                break
            
            # 智能停止条件检查（只在生成一定数量后检查）
            if step > 20:  # 至少生成20个token后再检查
                if self.content_filter.should_stop_generation(generated, self.model.tokenizer, step):
                    print(f"Smart stopping triggered, stop generating at step {step}")
                    break
            
            # 每个decode开始计时
            decode_start_time = time.time()
            next_token, kv_cache = self.session_mgr.run_decode(next_token, kv_cache, prompt_length + step - 1, generated)
            # 每个Decode结束计时
            decode_end_time = time.time()
            decode_times_ms.append((decode_end_time - decode_start_time) * 1000)
            generated.append(next_token)

            if step % 10 == 0:
                print(f"Generated {step}/{max_length} tokens...")
        
        # 生成结束时显示统计信息
        print(f"\n[生成完成，共生成 {len(generated)} 个token]")
        
        #--- 总结束计时
        total_end_time = time.time()
        total_time_ms = (total_end_time - total_start_time) * 1000

        # 打印总时间和平均输出速度
        total_time_s = total_time_ms / 1000.0
        print(f"Total time: {total_time_s:.2f} s")
        print(f"Average output speed: {len(generated) / total_time_s:.2f} tokens/s")
        # print(f"Prefill time: {prefill_time_ms:.2f} ms")
        print(f"Average decode time: {sum(decode_times_ms) / len(decode_times_ms):.2f} ms")
        
        # 解码结果
        # detokenizer_start_time = time.time()
        result = self.model.tokenizer.decode(generated)
        # detokenizer_end_time = time.time()
        # detokenizer_time_ms = (detokenizer_end_time - detokenizer_start_time) * 1000
        # print(f"{len(generated)} 个token文本解码耗时: {detokenizer_time_ms:.4f}ms")
        
        # 优化后处理，使用ContentFilter的智能清理
        result = self._post_process_result(result)

        return result

    def _post_process_result(self, result: str) -> str:
        """优化后处理，使用智能清理逻辑"""
        if not result:
            return result
        
        # 移除多余的空行和空白字符
        lines = [line.strip() for line in result.split('\n') if line.strip()]
        
        # 检查内容质量
        if len(lines) == 0:
            return ""
        
        # 简单的句子完整性检查
        cleaned_lines = []
        for line in lines:
            # 移除明显不完整的句子（如只有标点符号或单个词）
            if len(line) > 3 and not line.endswith(('，', '、', '、', '、')):
                cleaned_lines.append(line)
        
        # 合并为最终结果
        cleaned_result = '\n'.join(cleaned_lines)
        
        return cleaned_result
    
def main():
    print("=== Qwen2-Instruct Demo ===")
    # 创建生成器
    config = ModelConfig()
    generator = QwenGenerator(config)
    
    while True:
        try: 
            prompt = input("\n请输入问题(回车退出): ").strip()
            if not prompt:
                break
            print("Generating...")
            answer = generator.generate(prompt=prompt,max_length=config.max_new_tokens)
            print('-' * 80)
            print("Answer:", answer)
        except KeyboardInterrupt:
            print("\nUser interrupted")
            break
        except Exception as e:
            print(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            print("-"*80)
            continue

if __name__ == "__main__":
    main()