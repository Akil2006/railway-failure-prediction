async function runPrediction() {

    // UI elements
    const errorBox = document.getElementById("errorBox");
    const placeholder = document.getElementById("placeholder");
    const spinner = document.getElementById("spinner");
    const results = document.getElementById("results");

    errorBox.classList.add("hidden");
    placeholder.classList.add("hidden");
    results.classList.add("hidden");
    spinner.classList.remove("hidden");

    try {
        // Collect input values
        const data = {
            air_temperature: document.getElementById("air_temperature").value,
            process_temperature: document.getElementById("process_temperature").value,
            rotational_speed: document.getElementById("rotational_speed").value,
            torque: document.getElementById("torque").value,
            tool_wear: document.getElementById("tool_wear").value
        };

        // Call backend
        const response = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });

        const result = await response.json();
        console.log(result);

        spinner.classList.add("hidden");
        results.classList.remove("hidden");

        // ❌ Handle error
        if (result.error) {
            errorBox.innerText = result.error;
            errorBox.classList.remove("hidden");
            return;
        }

        // ✅ FINAL DECISION
        document.getElementById("verdictText").innerText =
            result.final.decision;

        document.getElementById("avgProb").innerText =
            (result.final.avg_probability * 100).toFixed(2) + "%";

        // ✅ LSTM OUTPUT
        document.getElementById("lstmLabel").innerText =
            result.lstm.label === 1 ? "Failure" : "No Failure";

        document.getElementById("lstmProb").innerText =
            (result.lstm.probability * 100).toFixed(2) + "%";

        document.getElementById("lstmConf").innerText =
            ((result.lstm.label === 1 ? result.lstm.probability : 1 - result.lstm.probability) * 100).toFixed(1) + "%";

        document.getElementById("lstmBar").style.width =
            (result.lstm.probability * 100) + "%";

        // ✅ GRU OUTPUT
        document.getElementById("gruLabel").innerText =
            result.gru.label === 1 ? "Failure" : "No Failure";

        document.getElementById("gruProb").innerText =
            (result.gru.probability * 100).toFixed(2) + "%";

        document.getElementById("gruConf").innerText =
            ((result.gru.label === 1 ? result.gru.probability : 1 - result.gru.probability) * 100).toFixed(1) + "%";

        document.getElementById("gruBar").style.width =
            (result.gru.probability * 100) + "%";

    } catch (error) {
        spinner.classList.add("hidden");
        errorBox.innerText = "⚠ Request failed: " + error.message;
        errorBox.classList.remove("hidden");
    }
}