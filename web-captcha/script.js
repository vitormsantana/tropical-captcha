// Function to divide string into chunks
function divideString(str, chunkSize) {
    const chunks = [];
    for (let i = 0; i < str.length; i += chunkSize) {
        chunks.push(str.slice(i, i + chunkSize));
    }
    return chunks.join('\n');
}

// Function to get base64 data from image
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

// Function to send JSON data to AWS API Gateway
async function sendBase64ToAPI(jsonData) {
    const apiUrl = "https://132fpvvrhi.execute-api.sa-east-1.amazonaws.com/cors/predict";
	
	console.log("Data sent to API:", jsonData);
	
    try {
        const response = await fetch(apiUrl, {
            method: "POST",
            body: JSON.stringify(jsonData),
            headers: {
                "Content-Type": "application/json"
            },
        });

        console.log("API Response Status:", response.status);

        // Return the entire response object
        return response;
    } catch (error) {
        console.error("An error occurred:", error);
    }
}

// DOMContentLoaded event handler
document.addEventListener("DOMContentLoaded", async () => {
    const captchaImage = document.getElementById("captcha-image");
    const predictButton = document.getElementById("predict-button");
    const resultContainer = document.getElementById("prediction-result");
    const base64Display = document.getElementById("base64-data");

    captchaImage.src = "http://127.0.0.1:8080/processed_p5d3tr.png";

    predictButton.addEventListener("click", async () => {
        try {
            const base64Captcha = await getBase64FromImage(captchaImage);
            console.log("Base64 Data:", base64Captcha);

        const requestData = {
            body: {
                data: `data:image/png;base64,${base64Captcha}`
            }
        };

            const response = await sendBase64ToAPI(requestData);

            if (response.ok) {
                console.log("Base64 data sent successfully.");
                const responseBody = await response.json();
                console.log("API Response Body:", responseBody);

                resultContainer.textContent = "Prediction: " + responseBody.body.prediction;

                // Display base64
                base64Display.textContent = "Base64 Data:\n" + divideString(base64Captcha, 120);
            } else {
                console.error("Error sending base64 data.");
            }
        } catch (error) {
            console.error("An error occurred:", error);
            resultContainer.textContent = "Error: Failed to predict";
        }
    });
});
