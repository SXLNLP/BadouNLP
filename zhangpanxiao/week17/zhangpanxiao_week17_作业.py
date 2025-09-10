import time
from typing import Dict, List, Optional

class DialogueSystem:
    def __init__(self):
        # 对话状态
        self.state = {
            'current_intent': None,
            'slots': {},
            'awaiting_input': False,
            'last_user_utterance': None
        }
        
        # 历史记录（存储最近的系统回复）
        self.history = {
            'system_responses': [],  # 存储系统回复的文本
            'audio_cache': None     # 可选：存储语音音频（这里用文本模拟）
        }
        
        # 模拟的天气数据
        self.weather_data = {
            '北京': {
                'today': {'condition': '晴', 'high': 32, 'low': 25},
                'tomorrow': {'condition': '多云', 'high': 30, 'low': 24}
            },
            '上海': {
                'today': {'condition': '小雨', 'high': 28, 'low': 23},
                'tomorrow': {'condition': '阴', 'high': 27, 'low': 22}
            }
        }
    
    def process_input(self, user_input: str):
        """处理用户输入的核心方法"""
        self.state['last_user_utterance'] = user_input
        
        # 1. 意图识别
        intent = self._understand_intent(user_input)
        self.state['current_intent'] = intent
        
        # 2. 处理重听意图（最高优先级）
        if intent == 'repeat':
            self._handle_repeat_request()
            return
        
        # 3. 正常业务流程
        if intent == 'query_weather':
            self._handle_weather_query()
        elif intent == 'greet':
            self._respond("你好！我是语音助手，可以问我天气信息。")
        else:
            self._respond("抱歉，我没有理解您的意思。")
    
    def _understand_intent(self, text: str) -> str:
        """简化的意图识别"""
        text = text.lower()
        
        # 重听意图检测
        repeat_keywords = ['再说一遍', '重复', '没听清', '没听到']
        if any(keyword in text for keyword in repeat_keywords):
            return 'repeat'
        
        # 其他意图
        if '天气' in text or '气温' in text:
            return 'query_weather'
        elif '你好' in text or '嗨' in text:
            return 'greet'
        else:
            return 'unknown'
    
    def _handle_repeat_request(self):
        """处理重听请求"""
        if not self.history['system_responses']:
            self._respond("没有可重复的内容。")
            return
        
        # 获取最后一次系统回复
        last_response = self.history['system_responses'][-1]
        
        # 模拟播放缓存的音频（实际应用中这里可能是真正的音频播放）
        print(f"[系统重播] {last_response}")
        
        # 保持当前状态不变
        self.state['awaiting_input'] = True
    
    def _handle_weather_query(self):
        """处理天气查询"""
        # 简化的槽位提取
        city, date = None, 'today'
        text = self.state['last_user_utterance']
        
        if '北京' in text:
            city = '北京'
        elif '上海' in text:
            city = '上海'
        
        if '明天' in text:
            date = 'tomorrow'
        
        if not city:
            self._respond("请问您想查询哪个城市的天气？")
            return
        
        # 获取天气数据
        weather = self.weather_data[city][date]
        response = (f"{city}{'明天' if date == 'tomorrow' else '今天'}天气{weather['condition']}，"
                   f"最高气温{weather['high']}度，最低气温{weather['low']}度。")
        
        self._respond(response)
    
    def _respond(self, text: str):
        """系统回复并缓存历史"""
        print(f"[系统] {text}")
        
        # 缓存系统回复
        self.history['system_responses'].append(text)
        # 在实际应用中，这里还会缓存音频文件
        # self.history['audio_cache'] = tts.generate_audio(text)
        
        # 更新状态
        self.state['awaiting_input'] = True
    
    def run(self):
        """运行对话系统"""
        print("对话系统已启动。输入'退出'结束对话。")
        
        while True:
            user_input = input("\n[用户] ")
            
            if user_input.lower() in ['退出', 'exit', 'quit']:
                print("对话结束。")
                break
                
            self.process_input(user_input)

# 运行示例
if __name__ == "__main__":
    system = DialogueSystem()
    system.run()
