document.addEventListener("DOMContentLoaded", async () => {
    const captchaImage = document.getElementById("captcha-image");
    const predictButton = document.getElementById("predict-button");
    const resultContainer = document.getElementById("result-container");
    const base64Display = document.getElementById("base64-data"); // Add this line

    captchaImage.src = "http://127.0.0.1:8080/processed_p5d3tr.PNG";

    predictButton.addEventListener("click", async () => {
        try {
            const base64Captcha = await getBase64FromImage(captchaImage);
            console.log("Base64 Data:", base64Captcha);

            const requestData = {
                data: base64Captcha
            };

            const response = await sendBase64ToAPI(requestData);

            const responseBody = await response.json();
            resultContainer.textContent = "Prediction: " + responseBody.prediction;

            // Display base64
            base64Display.textContent = "Base64 Data: " + base64Captcha;
        } catch (error) {
            console.error("An error occurred:", error);
            resultContainer.textContent = "Error: Failed to predict";
        }
    });
});

async function getBase64FromImage(imageElement) {
    return new Promise((resolve, reject) => {
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d");

        canvas.width = imageElement.width;
        canvas.height = imageElement.height;

        context.drawImage(imageElement, 0, 0, canvas.width, canvas.height);

        const base64Data = canvas.toDataURL("image/png").split(',')[1];

        resolve(base64Data);
    });
}

async function sendBase64ToAPI(data) {
    const apiUrl = "https://132fpvvrhi.execute-api.sa-east-1.amazonaws.com/cors/predict";

    try {
        const response = await fetch(apiUrl, {
            method: "POST",
            body: JSON.stringify(data),
            headers: {
                "Content-Type": "application/json"
            },
        });

        if (response.ok) {
            console.log("Base64 data sent successfully.");
            console.log("Response Body:", response);

            return response; // Return the response object
        } else {
            console.error("Error sending base64 data.");
        }
    } catch (error) {
        console.error("An error occurred:", error);
    }
}
