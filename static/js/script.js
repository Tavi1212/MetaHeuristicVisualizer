const discreteBtn = document.getElementById("problem-discrete");
const continuousBtn = document.getElementById("problem-continuous");

const discreteDiv = document.getElementById("discrete-options");
const continuousDiv = document.getElementById("continuous-options");

const discreteShannonBtn = document.getElementById("discrete-entropy");
const discreteClusteringBtn = document.getElementById("discrete-clustering");
const continuousStdBtn = document.getElementById("continuous-standard");
const continuousClusteringBtn = document.getElementById(
    "continuous-clustering"
);

const discreteShannon = document.getElementById("discrete-entropy-options");
const discreteClustering = document.getElementById(
    "discrete-clustering-options"
);
const continuousStd = document.getElementById("continuous-standard-options");
const continuousClustering = document.getElementById(
    "continuous-clustering-options"
);

discreteBtn.addEventListener("change", () => {
    if (discreteBtn.checked) {
        discreteDiv.style.display = "block";
        continuousDiv.style.display = "none";

        document
            .querySelectorAll("#discrete-options input")
            .forEach((el) => (el.disabled = false));
        document
            .querySelectorAll("#discrete-options select")
            .forEach((el) => (el.disabled = false));
        // Disable continuous inputs
        document
            .querySelectorAll("#continuous-options input")
            .forEach((el) => (el.disabled = true));
        document
            .querySelectorAll("#continuous-options select")
            .forEach((el) => (el.disabled = true));

        discreteShannonBtn.checked = true;
        discreteShannon.style.display = "block";
        discreteClustering.style.display = "none";
        continuousStd.style.display = "none";
        continuousClustering.style.display = "none";
    }
});

continuousBtn.addEventListener("change", () => {
    if (continuousBtn.checked) {
        continuousDiv.style.display = "block";
        discreteDiv.style.display = "none";

        document
            .querySelectorAll("#continuous-options input")
            .forEach((el) => (el.disabled = false));
        document
            .querySelectorAll("#continuous-options select")
            .forEach((el) => (el.disabled = false));

        document
            .querySelectorAll("#discrete-options input")
            .forEach((el) => (el.disabled = true));
        document
            .querySelectorAll("#discrete-options select")
            .forEach((el) => (el.disabled = true));

        continuousStdBtn.checked = true;
        discreteShannon.style.display = "none";
        discreteClustering.style.display = "none";
        continuousStd.style.display = "block";
        continuousClustering.style.display = "none";
    }
});

discreteShannonBtn.addEventListener("change", () => {
    if (discreteShannonBtn.checked) {
        discreteShannon.style.display = "block";
        discreteClustering.style.display = "none";
        continuousStd.style.display = "none";
        continuousClustering.style.display = "none";
    }
});

discreteClusteringBtn.addEventListener("change", () => {
    if (discreteClusteringBtn.checked) {
        discreteShannon.style.display = "none";
        discreteClustering.style.display = "block";
        continuousStd.style.display = "none";
        continuousClustering.style.display = "none";
    }
});

continuousStdBtn.addEventListener("change", () => {
    if (continuousStdBtn.checked) {
        discreteShannon.style.display = "none";
        discreteClustering.style.display = "none";
        continuousStd.style.display = "block";
        continuousClustering.style.display = "none";
    }
});

continuousClusteringBtn.addEventListener("change", () => {
    if (continuousClusteringBtn.checked) {
        discreteShannon.style.display = "none";
        discreteClustering.style.display = "none";
        continuousStd.style.display = "none";
        continuousClustering.style.display = "block";
    }
});

function displayAdvancedSettings() {
    console.log("advanced setting slider");

    let optionsTab = document.createElement("div");
}

function addAlgorithm() {
    console.log("algorithm button pressed");

    const index = document.querySelectorAll(".algorithm").length;

    let newAlgorithm = document.createElement("div");
    newAlgorithm.classList.add("algorithm");

    let nameInputBox = document.createElement("div");
    let nameLabel = document.createTextNode("Name: ");
    let nameInput = document.createElement("input");
    nameInputBox.classList.add("name-input-box");
    nameInput.type = "text";
    nameInput.name = `name_${index}`; // assign name for easier debugging

    let colorInputBox = document.createElement("div");
    let colorLabel = document.createTextNode("Color: ");
    let colorInput = document.createElement("input");
    colorInputBox.classList.add("color-input-box");
    colorInput.type = "color";
    colorInput.classList.add("color-picker");
    colorInput.name = `color_${index}`;

    let chooseFileBox = document.createElement("div");
    let chooseFile = document.createElement("input");
    chooseFileBox.classList.add("choose-file-box");
    chooseFile.type = "file";
    chooseFile.accept = ".txt, .csv";
    chooseFile.name = `file_${index}`;

    let removeButtonBox = document.createElement("div");
    let removeButton = document.createElement("button");
    removeButtonBox.classList.add("remove-button-box");
    removeButton.type = "button";
    removeButton.innerText = "Remove";
    removeButton.classList.add("remove-button");
    removeButton.onclick = function () {
        removeAlgorithmBtnClick(this);
    };

    nameInputBox.appendChild(nameLabel);
    nameInputBox.appendChild(nameInput);
    colorInputBox.appendChild(colorLabel);
    colorInputBox.appendChild(colorInput);
    chooseFileBox.appendChild(chooseFile);
    removeButtonBox.appendChild(removeButton);

    newAlgorithm.appendChild(nameInputBox);
    newAlgorithm.appendChild(colorInputBox);
    newAlgorithm.appendChild(chooseFileBox);
    newAlgorithm.appendChild(removeButtonBox);
    document.getElementById("algorithm_list_container").appendChild(newAlgorithm);
}

function logAlgorithmDetails() {
    const algorithms = document.querySelectorAll(".algorithm");
    const formData = new FormData();
    let valid = false;

    algorithms.forEach((algorithm, index) => {
        const nameInput = algorithm.querySelector(".name-input-box input");
        const colorInput = algorithm.querySelector(".color-input-box input");
        const fileInput = algorithm.querySelector(".choose-file-box input");

        if (
            nameInput &&
            nameInput.value.trim() !== "" &&
            fileInput.files.length > 0
        ) {
            valid = true;

            const name = nameInput.value;
            const color = colorInput.value;
            const file = fileInput.files[0];

            // Append data to formData
            formData.append(`name_${index}`, name);
            formData.append(`color_${index}`, color);
            formData.append(`file_${index}`, file);

            // Log info to console
            console.log(`Algorithm ${index + 1}:`);
            console.log(`  Name: ${name}`);
            console.log(`  Color: ${color}`);
            console.log(`  File: ${file.name}`);
        }
    });

    if (valid) {
        formData.append("total", algorithms.length);
        sendDataToFlask(formData);
    } else {
        alert("Please add at least one algorithm with a file");
    }
}

function sendDataToFlask(formData) {
    fetch("/create_graphs", {
        method: "POST",
        body: formData,
    })
        .then((response) => {
            if (response.ok) {
                window.location.href = "/";
            } else {
                console.error("Error creating graphs");
            }
        })
        .catch((error) => console.error("Error:", error));
}

function submitAllData() {
    const configForm = document.getElementById("problem-config-form");
    const configData = new FormData(configForm);

    const selectedStrategy = document.querySelector('input[name="strategy"]:checked');
    if (selectedStrategy) {
        configData.set("partitionStrategy", selectedStrategy.value);
        console.log("📤 Partition strategy set to:", selectedStrategy.value);
    }

    fetch("/submit", {
        method: "POST",
        body: configData,
    })
        .then((response) => {
            if (!response.ok) {
                throw new Error("Config submission failed");
            }
            return submitAlgorithmData();
        })
        .then(() => {
            const iframe = document.getElementById("graph-frame");
            iframe.src = "/display_graph/fr"; // default layout after generation
        })
        .catch((error) => {
            console.error("Error in submission chain:", error);
            alert(
                "There was a problem submitting your data. See console for details."
            );
        });
}

function submitAlgorithmData() {
    return new Promise((resolve, reject) => {
        const algorithms = document.querySelectorAll(".algorithm");
        const formData = new FormData();
        let valid = false;

        algorithms.forEach((algorithm, index) => {
            const nameInput = algorithm.querySelector(".name-input-box input");
            const colorInput = algorithm.querySelector(".color-input-box input");
            const fileInput = algorithm.querySelector(".choose-file-box input");

            if (
                nameInput &&
                nameInput.value.trim() !== "" &&
                fileInput.files.length > 0
            ) {
                valid = true;
                formData.append(`name_${index}`, nameInput.value);
                formData.append(`color_${index}`, colorInput.value);
                formData.append(`file_${index}`, fileInput.files[0]);
            }
        });

        if (!valid) {
            alert("Please add at least one algorithm with a file");
            reject("No valid algorithms");
            return;
        }

        formData.append("total", algorithms.length);

        fetch("/generate_visualization", {
            method: "POST",
            body: formData,
        })
            .then((response) => {
                if (response.ok) {
                    resolve();
                } else {
                    reject("Algorithm submission failed");
                }
            })
            .catch(reject);
    });
}

var iframe = document.getElementById("graph-frame");
iframe.addEventListener("load", function () {
    var doc = iframe.contentDocument || iframe.contentWindow.document;
    // Example: inject CSS to force the graph to fill the iframe
    var style = doc.createElement("style");
    style.innerHTML = `
        * { margin: 0; padding:0}
        body {overflow-y: hidden;}
        .card{height:100%}
        #loadingBar{ 
        position: absolute; 
        inset: 0 !important;
         width: 100vw !important; 
         height: 100vh !important}
        #mynetwork{ height:100% !important;}
    `;
    doc.head.appendChild(style);

    // Now show the iframe
    iframe.style.visibility = "visible";
});

function toggleCollapsible(header) {
    const section = header.closest(".collapsible-section");
    section.classList.toggle("collapsed");
}

function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    sidebar.classList.toggle("collapsed");
}

function removeAlgorithmBtnClick(elt) {
    const container = document.getElementById("algorithm_list_container");
    if (container.children.length > 1) {
        elt.closest(".algorithm").remove();
    }
}

function downloadFormAsJSON() {
    const form = document.getElementById("problem-config-form");
    const formData = new FormData(form);
    // Also include unchecked checkboxes and radio buttons
    form.querySelectorAll("input[type=checkbox]:not(:checked)").forEach((cb) => {
        if (!formData.has(cb.name)) formData.set(cb.name, false);
    });
    form.querySelectorAll("input[type=checkbox]:checked").forEach((cb) => {
        formData.set(cb.name, true);
    });

    // For radio buttons, ensure only the checked value is included
    form.querySelectorAll("input[type=radio]").forEach((rb) => {
        if (rb.checked) formData.set(rb.name, rb.value);
    });

    // For selects, get their value
    form.querySelectorAll("select").forEach((sel) => {
        formData.set(sel.name, sel.value);
    });

    // For all inputs, get their value (including text, number, range)
    form
        .querySelectorAll("input[type=text],input[type=number],input[type=range]")
        .forEach((inp) => {
            formData.set(inp.name, inp.value);
        });

    // Build JSON object
    const obj = {};
    for (const [key, value] of formData.entries()) {
        obj[key] = value;
    }

    // Download as JSON file
    const blob = new Blob([JSON.stringify(obj, null, 2)], {
        type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "config.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function uploadFormFromJSON() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "application/json";
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const text = await file.text();
        let data;
        try {
            data = JSON.parse(text);
        } catch (err) {
            alert("Invalid JSON file.");
            return;
        }
        const form = document.getElementById("problem-config-form");
        for (const [key, value] of Object.entries(data)) {
            const elements = form.querySelectorAll(`[name="${key}"]`);
            elements.forEach((el) => {
                if (el.type === "checkbox") {
                    el.checked = value === true || value === "true";
                } else if (el.type === "radio") {
                    el.checked = el.value == value;
                } else if (el.tagName === "SELECT") {
                    el.value = value;
                } else {
                    el.value = value;
                }
                // Trigger input events for live updates (e.g., range display)
                el.dispatchEvent(new Event("input", {bubbles: true}));
                el.dispatchEvent(new Event("change", {bubbles: true}));
            });
        }
    };
    input.click();
}

function switchLayout(layout) {
    const treeLayoutCheckbox = document.getElementById("tree_layout");
    if (treeLayoutCheckbox && treeLayoutCheckbox.checked) {
        alert("Tree layout is already applied. Please disable it to use other layouts.");
        // Optionally reset selection
        document.getElementById(layout === 'fr' ? 'kk-layout' : 'fr-layout').checked = true;
        return;
    }

    // Update iframe source
    const iframe = document.getElementById("graph-frame");
    iframe.src = `/display_graph/${layout}`;
}

async function downloadPDF() {
  const iframe = document.getElementById("graph-frame");
  const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
  const content = iframeDoc.body;

  // Wait a little to ensure everything is rendered
  await new Promise(resolve => setTimeout(resolve, 500));

  html2canvas(content, { useCORS: true }).then(canvas => {
    const imgData = canvas.toDataURL("image/png");
    const pdf = new jspdf.jsPDF({
      orientation: 'landscape',
      unit: 'px',
      format: [canvas.width, canvas.height]
    });
    pdf.addImage(imgData, 'PNG', 0, 0, canvas.width, canvas.height);
    pdf.save("stn_graph.pdf");
  });
}