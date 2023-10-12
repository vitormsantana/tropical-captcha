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

//function to randomly select an image to predict
function getRandomImageName() {
    const imageNames = [
        "captcha_0",
		"captcha_1",
		"captcha_2",
		"captcha_4",
		"captcha_5",
		"captcha_6",
		"captcha_7",
		"captcha_8",
		"captcha_9",
		"captcha_10",
		"captcha_11"
        /*"processed_captcha_002.png",
        "processed_captcha_003.png",
		"processed_captcha_004.png",
        "processed_captcha_005.png",
        "processed_captcha_006.png",
		"processed_captcha_007.png",
        "processed_captcha_008.png",
        "processed_captcha_009.png",
		"processed_captcha_010.png",
        "processed_captcha_011.png",
        "processed_captcha_012.png",
		"processed_captcha_013.png",
        "processed_captcha_014.png",
        "processed_captcha_015.png",
		*/
        // Add more image names as needed
    ];

    const randomIndex = Math.floor(Math.random() * imageNames.length);
    return imageNames[randomIndex];
}


document.addEventListener("DOMContentLoaded", async () => {
    const captchaImage = document.getElementById("captcha-image");
    const predictButton = document.getElementById("predict-button");
    const nextButton = document.getElementById("next-button"); // Add this line
    const resultContainer = document.getElementById("prediction-result");
    const base64Display = document.getElementById("base64-data");

	predictButton.addEventListener("click", async () => {
		try {
			const base64Captcha = await getBase64FromImage(captchaImage);

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

				const bodyJSON = JSON.parse(responseBody.body);
				const prediction = bodyJSON.prediction;

				// Set the prediction text within the styled container
				resultContainer.textContent = prediction;
				resultContainer.style.display = "block"; // Make the container visible

				// Display base64
				//base64Display.textContent = "Base64 Data:\n" + divideString(base64Captcha, 120);
			} else {
				console.error("Error sending base64 data.");
			}
		} catch (error) {
			console.error("An error occurred:", error);
			resultContainer.textContent = "Error: Failed to predict";
		}
});

    nextButton.addEventListener("click", async () => {
        try {
            const randomImageName = getRandomImageName();
            const imageSrc = `./${randomImageName}.png`;

            captchaImage.src = imageSrc;

            resultContainer.textContent = "Prediction: "; // Clear the prediction
            //base64Display.textContent = "Base64 Data:\n-"; // Clear the base64 data
        } catch (error) {
            console.error("An error occurred:", error);
        }
    });
});

