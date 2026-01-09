"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { Send, Menu, Plus, MessageSquare } from "lucide-react";
import clsx from "clsx";

type Message = {
  role: "user" | "assistant";
  content: string;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      // Using direct URL to bypass Next.js proxy buffering potential
      const response = await fetch("http://localhost:8000/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: [...messages, userMessage],
          max_tokens: 512,
          temperature: 0.7,
        }),
      });

      if (!response.ok) throw new Error("Network response was not ok");
      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = "";

      // Add a placeholder assistant message
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        assistantMessage += text;

        setMessages((prev) => {
          const newMessages = [...prev];
          if (newMessages.length > 0) {
            const lastIdx = newMessages.length - 1;
            newMessages[lastIdx] = { ...newMessages[lastIdx], content: assistantMessage };
          }
          return newMessages;
        });
      }
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, something went wrong." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-[#212121]">
      {/* Sidebar - Darker */}
      <div
        className={clsx(
          "bg-[#171717] text-[#ECECEC] w-[260px] flex-shrink-0 transition-all duration-300 flex flex-col p-3",
          isSidebarOpen ? "translate-x-0" : "-translate-x-full absolute h-full z-10"
        )}
      >
        <div className="flex justify-between items-center mb-4">
          <button
            onClick={() => setMessages([])}
            className="flex-1 flex items-center gap-2 px-3 py-2 border border-white/20 rounded-md hover:bg-[#212121] transition-colors text-sm"
          >
            <Plus size={16} />
            New chat
          </button>
          <button
            onClick={() => setIsSidebarOpen(false)}
            className="ml-2 p-2 hover:bg-[#212121] rounded-md md:hidden"
          >
            <Menu size={20} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          <div className="text-xs font-semibold text-gray-500 mb-2 px-2">Today</div>
          <button className="flex items-center gap-3 px-3 py-2 w-full text-left hover:bg-[#212121] rounded-md transition-colors text-sm truncate text-[#ECECEC]">
            <MessageSquare size={16} />
            <span className="truncate">New conversation</span>
          </button>
        </div>

        {/* User Profile Area (Mock) */}
        <div className="pt-2 border-t border-white/10 mt-2">
          <button className="flex items-center gap-3 px-3 py-3 w-full hover:bg-[#212121] rounded-md transition-colors text-sm font-medium">
            <div className="w-8 h-8 rounded-full bg-green-700 flex items-center justify-center text-white">
              QP
            </div>
            <div className="flex-1 text-left">User</div>
          </button>
        </div>
      </div>

      {/* Main Chat Area - #212121 */}
      <div className="flex-1 flex flex-col h-full relative">
        {/* Top Bar (Mobile) / Menu Toggle */}
        <div className="absolute top-0 left-0 p-2 z-20 flex items-center gap-2">
          {!isSidebarOpen && (
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="p-2 text-gray-400 hover:text-white hover:bg-[#2F2F2F] rounded-md"
            >
              <Menu size={24} />
            </button>
          )}
          <div className="text-lg font-medium text-gray-200 md:hidden">LocalGPT 5.2</div>
        </div>


        {/* Chat History */}
        <div className="flex-1 overflow-y-auto w-full pt-12 md:pt-0">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-[#ECECEC]">
              <div className="w-12 h-12 bg-white text-black rounded-full flex items-center justify-center mb-4">
                {/* Simple logo placeholder */}
                <div className="w-6 h-6 border-2 border-black rounded-full"></div>
              </div>
              <h1 className="text-2xl font-semibold mb-8">How can I help you today?</h1>
            </div>
          ) : (
            <div className="flex flex-col pb-40 px-4 md:px-0">
              <div className="max-w-3xl mx-auto w-full flex flex-col gap-6 py-6">
                {messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={clsx(
                      "w-full flex",
                      msg.role === "assistant" ? "justify-start" : "justify-end"
                    )}
                  >
                    <div className={clsx(
                      "relative max-w-[85%] md:max-w-[80%] rounded-2xl px-5 py-3 text-base",
                      msg.role === "user"
                        ? "bg-[#2F2F2F] text-white rounded-br-sm"
                        : "text-white/90 pl-0"
                    )}>
                      <div className="prose prose-invert max-w-none">
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area - Floating Pill */}
        <div className="w-full pt-2 pb-6 px-4">
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleSubmit} className="relative w-full flex flex-col bg-[#2F2F2F] rounded-[26px] border border-[#424242] focus-within:border-gray-500 shadow-md overflow-hidden">
              <input
                className="w-full py-4 pl-4 pr-12 bg-transparent text-[#ECECEC] outline-none placeholder-gray-400 resize-none"
                placeholder="Message LocalGPT"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                autoComplete="off"
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className={clsx(
                  "absolute right-2 bottom-2 p-2 rounded-full transition-colors",
                  input.trim() ? "bg-white text-black hover:bg-gray-200" : "bg-[#676767] text-[#2F2F2F] cursor-not-allowed"
                )}
              >
                <Send size={16} />
              </button>
            </form>
            <div className="text-center text-xs text-gray-500 mt-2">
              LocalGPT can make mistakes. Check important info.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
