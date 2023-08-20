document.addEventListener("DOMContentLoaded", () => {
    const captchaImage = document.getElementById("captcha-image");
    const captchaInput = document.getElementById("captcha-input");
    const verifyButton = document.getElementById("verify-button");
    const resultMessage = document.getElementById("result-message");

    // Load captcha image from your local server
    captchaImage.src = "http://localhost:8000/captcha2564linesi.PNG"; // Replace with your image file name

    verifyButton.addEventListener("click", () => {
        const userInput = captchaInput.value;
        // Use your Keras model to solve the captcha and get the expected solution
        const expectedSolution = "i"; // Replace with your actual expected solution

        if (userInput === expectedSolution) {
            resultMessage.textContent = "Captcha solved correctly!";
        } else {
            resultMessage.textContent = "Captcha solution is incorrect.";
        }
    });
});

