class DialogueSystem:
    def __init__(self):
        # 存储对话历史
        self.dialogue_history = []
        # 重听关键词
        self.repeat_keywords = ["重复", "再说一遍", "没听清", "重听", "repeat"]

    def process_input(self, user_input):
        """处理用户输入"""
        # 检查是否是重听请求
        if any(keyword in user_input for keyword in self.repeat_keywords):
            return self.repeat_last_response()

        # 正常对话处理流程
        response = self.generate_response(user_input)

        # 保存到对话历史
        self.dialogue_history.append({
            "user": user_input,
            "system": response
        })

        return response

    def generate_response(self, user_input):
        """生成对话响应（示例逻辑）"""
        # 这里可以替换为实际的对话模型或逻辑
        if "你好" in user_input:
            return "你好！我是对话助手，有什么可以帮您的？"
        elif "天气" in user_input:
            return "今天天气晴朗，气温25度，适合外出。"
        elif "时间" in user_input:
            return f"现在是{datetime.now().strftime('%H:%M')}。"
        else:
            return "抱歉，我不太明白您的意思。您可以问我关于天气、时间等问题。"

    def repeat_last_response(self):
        """重听上一轮系统回复"""
        if len(self.dialogue_history) == 0:
            return "目前还没有对话历史可供重听。"

        last_response = self.dialogue_history[-1]["system"]
        return f"（重听）{last_response}"

    def run(self):
        """运行对话系统"""
        print("对话系统已启动，输入'exit'退出程序")

        while True:
            user_input = input("用户: ").strip()

            if user_input.lower() == 'exit':
                print("再见！")
                break

            response = self.process_input(user_input)
            print(f"系统: {response}")


# 运行对话系统
if __name__ == "__main__":
    from datetime import datetime

    system = DialogueSystem()
    system.run()
