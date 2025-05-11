document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const questionButtons = document.querySelectorAll('.question-btn');
    
    // Function to add a message to the chat
    function addMessage(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(isUser ? 'user' : 'bot');
        
        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        
        const messageText = document.createElement('p');
        messageText.textContent = message;
        
        messageContent.appendChild(messageText);
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to the bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to show typing indicator
    function showTypingIndicator() {
        const indicatorDiv = document.createElement('div');
        indicatorDiv.classList.add('message', 'bot', 'typing-indicator-container');
        
        const indicator = document.createElement('div');
        indicator.classList.add('typing-indicator');
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            indicator.appendChild(dot);
        }
        
        indicatorDiv.appendChild(indicator);
        chatMessages.appendChild(indicatorDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return indicatorDiv;
    }
    
    // Function to handle sending a message
    async function sendMessage(message) {
        if (!message.trim()) return;
        
        // Add user message to chat
        addMessage(message, true);
        
        // Clear input field
        userInput.value = '';
        
        // Show typing indicator
        const indicator = showTypingIndicator();
        
        try {
            // Send message to backend
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: message })
            });
            
            const data = await response.json();
            
            // Remove typing indicator
            indicator.remove();
            
            // Add bot response to chat
            addMessage(data.response);
        } catch (error) {
            // Remove typing indicator
            indicator.remove();
            
            // Show error message
            addMessage('Sorry, I encountered an error. Please try again.');
            console.error('Error sending message:', error);
        }
    }
    
    // Handle form submission
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const message = userInput.value;
        sendMessage(message);
    });
    
    // Handle suggested question clicks
    questionButtons.forEach(button => {
        button.addEventListener('click', function() {
            const question = button.textContent;
            userInput.value = question;
            sendMessage(question);
        });
    });

    // Focus input field on page load
    userInput.focus();
});