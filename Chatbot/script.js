const chatContainer = document.querySelector(".conversation");
const userInput = document.querySelector(".message-input");
const sendBtn = document.querySelector(".send-button");

function appendMessage(text, role) {
      const div = document.createElement("div");
      div.classList.add("message", role);
      div.textContent = text;
      chatContainer.appendChild(div);
      chatContainer.scrollTop = chatContainer.scrollHeight;
      return div;
    }


    sendBtn.onclick = async () => {
      const question = userInput.value.trim();
      if (!question) return;

      appendMessage(question, "user");
      const botMsg = appendMessage("", "bot");

      userInput.value = "";

      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let done = false;
      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        const chunk = decoder.decode(value);
        botMsg.textContent += chunk;
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }
    };
    
    
    userInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault(); // prevent newline in input if multiline
    sendBtn.click();        // trigger send button click
  }
});
