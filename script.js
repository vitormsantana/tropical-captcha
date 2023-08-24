// Loading the Captcha Image and Sending Base64 Data: //

document.addEventListener("DOMContentLoaded", async () => {
    const captchaImage = document.getElementById("captcha-image");
    const predictButton = document.getElementById("predict-button");
    const resultContainer = document.getElementById("result-container");

    captchaImage.src = "https://vitormsantana.github.io/tropical-captcha/processed_p5d3tr.PNG";


    predictButton.addEventListener("click", async () => {
        try {
            const base64Captcha = await getBase64FromImage(captchaImage);

            // Create a JSON object with the base64 data
            const requestData = {
                data: base64Captcha
            };

            // Send the JSON object to your AWS REST API
            const response = await sendBase64ToAPI(requestData);

            // Display the prediction result
            const responseBody = await response.json();
            resultContainer.textContent = "Prediction: " + responseBody.prediction;
        } catch (error) {
            console.error("An error occurred:", error);
            resultContainer.textContent = "Error: Failed to predict";
        }
    });
});

// Extracting Base64 Data from Image: //

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


// Sending Base64 Data to AWS API://

async function sendBase64ToAPI(data) {
    const apiUrl = "https://132fpvvrhi.execute-api.sa-east-1.amazonaws.com/test"; // Replace with your AWS API endpoint URL

    try {
        const response = await fetch(apiUrl, {
            //mode: 'no-cors',
			//credentials: 'same-origin',
			method: "POST",
            body: JSON.stringify(data),
            headers: {
                "Content-Type": "application/json"
            }
        });

        if (response.ok) {
            console.log("Base64 data sent successfully.");
        } else {
            console.error("Error sending base64 data.");
        }
    } catch (error) {
        console.error("An error occurred:", error);
    }
}
