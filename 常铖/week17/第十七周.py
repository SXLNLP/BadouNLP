import React, { useState } from 'react';

function ChatBot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const sendMessage = () => {
    if (input.trim()) {
      setMessages([...messages, { sender: 'user', text: input }]);
      setInput('');
      // Simulate a bot response
      setTimeout(() => {
        setMessages(prevMessages => [...prevMessages, { sender: 'bot', text: '这是机器人的回复' }]);
      }, 1000);
    }
  };

  const replayConversation = () => {
    alert('重播对话');
    // Here you can add logic to replay the conversation
    console.log(messages);
  };

  return (
    <div style={{ width: '300px', margin: 'auto', border: '1px solid #ccc', padding: '10px', borderRadius: '5px' }}>
      <h2>聊天机器人</h2>
      <div style={{ height: '200px', overflowY: 'scroll', marginBottom: '10px' }}>
        {messages.map((msg, index) => (
          <div key={index} style={{ textAlign: msg.sender === 'user' ? 'right' : 'left', marginBottom: '5px' }}>
            <span style={{ backgroundColor: msg.sender === 'user' ? '#dcf8c6' : '#e0e0e0', padding: '5px', borderRadius: '5px', display: 'inline-block' }}>
              {msg.text}
            </span>
          </div>
        ))}
      </div>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
        placeholder="输入消息..."
        style={{ width: 'calc(100% - 70px)', marginRight: '5px', padding: '5px', boxSizing: 'border-box' }}
      />
      <button onClick={sendMessage} style={{ padding: '5px 10px' }}>发送</button>
      <br />
      <button onClick={replayConversation} style={{ marginTop: '10px', padding: '5px 10px' }}>重听</button>
    </div>
  );
}

export default ChatBot;
