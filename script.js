document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');

    let sessionId = localStorage.getItem('ragChatSessionId');
    if (!sessionId) {
        sessionId = generateUUID();
        localStorage.setItem('ragChatSessionId', sessionId);
    }
    console.log("Current Session ID:", sessionId);

    // --- Helper Functions ---
    function generateUUID() { // Provided by RFC4122 via a simple Math.random based approach
        let d = new Date().getTime(); //Timestamp
        let d2 = (performance && performance.now && (performance.now() * 1000)) || 0; //Time in microseconds since page-load or 0 if unsupported
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            let r = Math.random() * 16; //random number between 0 and 16
            if (d > 0) { //Use timestamp until depleted
                r = (d + r) % 16 | 0;
                d = Math.floor(d / 16);
            } else { //Use microseconds since page-load if supported
                r = (d2 + r) % 16 | 0;
                d2 = Math.floor(d2 / 16);
            }
            return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
        });
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendMessage(text, sender, isStreaming = false, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        
        const paragraph = document.createElement('p');
        paragraph.textContent = text;
        messageDiv.appendChild(paragraph);

        if (isStreaming) {
            messageDiv.classList.add('typing'); // Add typing class for initial AI placeholder
        }
        if (isError) {
            messageDiv.classList.add('error-message'); // General error styling
             // For AI streaming errors, the 'typing' class might still be on the div.
            messageDiv.classList.remove('typing');
        }
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
        return messageDiv; // Return the div for potential updates (streaming)
    }

    function updateStreamingMessage(aiMessageDiv, newContentChunk) {
        if (!aiMessageDiv) return;
        const paragraph = aiMessageDiv.querySelector('p');
        if (paragraph) {
            paragraph.textContent += newContentChunk;
        }
        scrollToBottom();
    }
    
    function finalizeStreamingMessage(aiMessageDiv) {
        if (aiMessageDiv) {
            aiMessageDiv.classList.remove('typing');
        }
    }

    function displayError(errorMessage, aiMessageDiv = null) {
        console.error("Error:", errorMessage);
        if (aiMessageDiv) { // If there's an AI placeholder, update it to show the error
            aiMessageDiv.classList.remove('typing');
            aiMessageDiv.classList.add('error-message'); // Add specific error styling
            const paragraph = aiMessageDiv.querySelector('p');
            if (paragraph) {
                paragraph.textContent = `Error: ${errorMessage}`;
            }
        } else { // Otherwise, append a new error message bubble
            appendMessage(`Error: ${errorMessage}`, 'ai', false, true);
        }
    }
    
    function autoGrowTextarea(element) {
        element.style.height = 'auto'; // Reset height to shrink if text is deleted
        element.style.height = (element.scrollHeight) + 'px'; // Set to content height
        sendButton.disabled = element.value.trim() === ""; // Disable send if empty
    }

    // --- Event Listeners ---
    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // Prevent new line
            sendMessage();
        }
    });
    messageInput.addEventListener('input', () => {
        autoGrowTextarea(messageInput);
    });

    // Initialize button state and textarea height
    autoGrowTextarea(messageInput); 

    // --- Send Message Function ---
    async function sendMessage() {
        const queryText = messageInput.value.trim();
        if (queryText === "") return;

        sendButton.disabled = true;
        messageInput.value = "";
        autoGrowTextarea(messageInput); // Reset height after clearing

        appendMessage(queryText, 'user');
        const aiMessagePlaceholderDiv = appendMessage("Thinking...", 'ai', true);

        try {
            const response = await fetch('http://localhost:8000/query/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream'
                },
                body: JSON.stringify({
                    query_text: queryText,
                    session_id: sessionId
                })
            });

            if (!response.ok) { // Handle HTTP errors (e.g., 4xx, 5xx) before streaming starts
                const errorData = await response.json().catch(() => ({ detail: "Unknown HTTP error" }));
                throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let firstChunkReceived = false;
            let currentAIMessageContent = ""; // To build up the AI message content

            // Clear the "Thinking..." placeholder text once stream starts or first token arrives
            if (aiMessagePlaceholderDiv.querySelector('p')) {
                aiMessagePlaceholderDiv.querySelector('p').textContent = ""; 
            }

            while (true) {
                const { value, done } = await reader.read();
                if (done) {
                    finalizeStreamingMessage(aiMessagePlaceholderDiv);
                    console.log("Stream finished.");
                    break;
                }

                const chunk = decoder.decode(value, { stream: true });
                // Process potentially multiple SSE messages in a single chunk
                const lines = chunk.split('\n\n');

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const jsonDataStr = line.substring(5).trim(); // Length of "data: "
                        if (jsonDataStr) {
                            try {
                                const jsonData = JSON.parse(jsonDataStr);
                                if (jsonData.token) {
                                    if (!firstChunkReceived) {
                                        finalizeStreamingMessage(aiMessagePlaceholderDiv); // Remove 'typing'
                                        firstChunkReceived = true;
                                    }
                                    currentAIMessageContent += jsonData.token;
                                    updateStreamingMessage(aiMessagePlaceholderDiv, jsonData.token);
                                }
                            } catch (e) {
                                console.warn("Could not parse JSON from data line:", jsonDataStr, e);
                            }
                        }
                    } else if (line.startsWith("event: error")) {
                        const errorDataLine = lines.find(l => l.startsWith("data: "));
                        let errorDetail = "An error occurred during streaming.";
                        if (errorDataLine) {
                            const errorJsonStr = errorDataLine.substring(5).trim();
                            try {
                                const errorJson = JSON.parse(errorJsonStr);
                                errorDetail = errorJson.token || errorDetail; // Assuming error detail is in 'token'
                            } catch (e) {
                                console.warn("Could not parse JSON from error data line:", errorJsonStr, e);
                            }
                        }
                        console.error("SSE Error Event:", errorDetail);
                        displayError(errorDetail, aiMessagePlaceholderDiv);
                        finalizeStreamingMessage(aiMessagePlaceholderDiv);
                        return; // Stop processing further chunks on SSE error
                    }
                }
            }
        } catch (error) {
            console.error('Failed to send message or process stream:', error);
            displayError(error.message || "Failed to connect or unknown error.", aiMessagePlaceholderDiv);
            finalizeStreamingMessage(aiMessagePlaceholderDiv); // Ensure typing is removed on error
        } finally {
            sendButton.disabled = false;
            messageInput.focus();
            autoGrowTextarea(messageInput); // Recalculate in case of content change or focus
        }
    }
});
```
