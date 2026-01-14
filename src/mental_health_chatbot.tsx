import React, { useState, useRef, useEffect } from "react";

interface Message {
  sender: "user" | "bot";
  text: string;
}

const MentalHealthChatbot: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Call backend API
  const callMLModel = async (userMessage: string | File, type: "text" | "audio") => {
    setIsLoading(true);
    try {
    let response: Response | undefined;

      const formData = new FormData();
      if (type === "text") {
        formData.append("text", userMessage as string);
        response = await fetch("http://127.0.0.1:8000/predict_text", {
          method: "POST",
          body: formData,
        });
      } else {
        formData.append("audio", userMessage as File);
        response = await fetch("http://127.0.0.1:8000/predict_audio", {
          method: "POST",
          body: formData,
        });
      }

      const data = await response.json();
      setMessages(prev => [...prev, { sender: "bot", text: data.prediction }]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { sender: "bot", text: "Error connecting to backend." }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendText = (text: string) => {
    if (!text.trim()) return;
    setMessages(prev => [...prev, { sender: "user", text }]);
    callMLModel(text, "text");
  };

  const handleAudioUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setMessages(prev => [...prev, { sender: "user", text: `Audio uploaded: ${file.name}` }]);
      callMLModel(file, "audio");
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>, text: string, setText: React.Dispatch<React.SetStateAction<string>>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendText(text);
      setText("");
    }
  };

  const [inputText, setInputText] = useState("");

  return (
    <div className="flex flex-col h-screen max-w-md mx-auto p-4 border rounded shadow">
      <div className="flex-1 overflow-y-auto mb-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`my-1 p-2 rounded ${msg.sender === "user" ? "bg-purple-200 self-end" : "bg-gray-200 self-start"}`}>
            {msg.text}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="flex gap-2">
        <textarea
          value={inputText}
          onChange={e => setInputText(e.target.value)}
          onKeyDown={e => handleKeyPress(e, inputText, setInputText)}
          placeholder="Type your message..."
          rows={2}
          className="flex-1 resize-none rounded border px-3 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
          disabled={isLoading}
        />
        <button
          onClick={() => {
            handleSendText(inputText);
            setInputText("");
          }}
          className="bg-purple-500 text-white px-4 py-2 rounded disabled:opacity-50"
          disabled={isLoading}
        >
          Send
        </button>
        <input type="file" accept="audio/*" onChange={handleAudioUpload} className="hidden" id="audio-upload" />
        <label htmlFor="audio-upload" className="bg-green-500 text-white px-4 py-2 rounded cursor-pointer">
          Upload Audio
        </label>
      </div>
    </div>
  );
};

export default MentalHealthChatbot;
fetch