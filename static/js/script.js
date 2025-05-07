const discreteBtn             = document.getElementById('discrete-btn');
const continuousBtn           = document.getElementById('continuous-btn');

const discreteDiv             = document.getElementById('discrete-opt');
const continuousDiv           = document.getElementById('continuous-opt');

const discreteShannonBtn      = document.getElementById('shannon-entropy-btn');
const discreteClusteringBtn   = document.getElementById('discrete-clustering-btn');
const continuousStdBtn        = document.getElementById('standard-partitioning-btn');
const continuousClusteringBtn = document.getElementById('continuous-clustering-btn');

const discreteShannon         = document.getElementById('discrete-shannon-opt');
const discreteClustering      = document.getElementById('discrete-clustering-opt');
const continuousStd           = document.getElementById('continuous-std-opt');
const continuousClustering    = document.getElementById('continuous-clustering-opt');


discreteBtn.addEventListener('change', () => {
    if (discreteBtn.checked) {
        discreteDiv.style.display = 'block';
        continuousDiv.style.display = 'none';

        discreteShannonBtn.checked = true;

        discreteShannon.style.display = 'block';
        discreteClustering.style.display = 'none';
        continuousStd.style.display = 'none';
        continuousClustering.style.display = 'none';
    }
});

continuousBtn.addEventListener('change', () => {
    if (continuousBtn.checked) {
        continuousDiv.style.display = 'block';
        discreteDiv.style.display = 'none';

        continuousStdBtn.checked = true;

        discreteShannon.style.display = 'none';
        discreteClustering.style.display = 'none';
        continuousStd.style.display = 'block';
        continuousClustering.style.display = 'none';
    }
});

discreteShannonBtn.addEventListener('change', () => {
    if(discreteShannonBtn.checked) {
        discreteShannon.style.display = 'block';
        discreteClustering.style.display = 'none';
        continuousStd.style.display = 'none';
        continuousClustering.style.display = 'none';
    }
});

discreteClusteringBtn.addEventListener('change', () => {
    if(discreteClusteringBtn.checked) {
        discreteShannon.style.display = 'none';
        discreteClustering.style.display = 'block';
        continuousStd.style.display = 'none';
        continuousClustering.style.display = 'none';
    }
});

continuousStdBtn.addEventListener('change', () => {
    if(continuousStdBtn.checked) {
        discreteShannon.style.display = 'none';
        discreteClustering.style.display = 'none';
        continuousStd.style.display = 'block';
        continuousClustering.style.display = 'none';
    }
});

continuousClusteringBtn.addEventListener('change', () => {
    if(continuousClusteringBtn.checked) {
        discreteShannon.style.display = 'none';
        discreteClustering.style.display = 'none';
        continuousStd.style.display = 'none';
        continuousClustering.style.display = 'block';
    }
});


function renderPDF(pdfUrl) {
  const canvas = document.getElementById("pdf-canvas");
  const context = canvas.getContext("2d");

  // Get the parent container's width dynamically
  const container = document.getElementById("pdf-viewer-container");
  const containerHeight = container.offsetHeight;

  // Load the PDF
  pdfjsLib.getDocument(pdfUrl).promise.then(function (pdfDoc_) {
    const pdfDoc = pdfDoc_;
    const pageNumber = 1; // You can loop through pages if needed

    // Fetch the page
    pdfDoc.getPage(pageNumber).then(function (page) {
      const viewport = page.getViewport({ scale: containerHeight / page.getViewport({ scale: 1 }).height });

      // Set the canvas size to match the viewport size
      canvas.width = viewport.width;
      canvas.height = containerHeight; // Set the canvas width to the container's width

      // Render the page into the canvas
      const renderContext = {
        canvasContext: context,
        viewport: viewport
      };
      page.render(renderContext);
    });
  });
}

function displayAdvancedSettings() {
    console.log("advanced setting slider");

    let optionsTab = document.createElement("div");
    
}

function addAlgorithm() {
    console.log("algorithm button pressed");

    let newAlgorithm = document.createElement("div");
    newAlgorithm.classList.add("algorithm");

    let nameInputBox = document.createElement("div");
    let nameLabel = document.createTextNode("Name: ");
    let nameInput = document.createElement("input");
    nameInputBox.classList.add("name-input-box");
    nameInput.type = "text";

    let colorInputBox = document.createElement("div");
    let colorLabel = document.createTextNode("Color: ");
    let colorInput = document.createElement("input");
    colorInputBox.classList.add("color-input-box");
    colorInput.type = "color";
    colorInput.classList.add("color-picker");

    let chooseFileBox = document.createElement("div");
    let chooseFile = document.createElement("input");
    chooseFileBox.classList.add("choose-file-box");
    chooseFile.type = "file";
    chooseFile.accept = ".txt, .csv";

    let removeButtonBox = document.createElement("div");
    let removeButton = document.createElement("button");
    removeButtonBox.classList.add("remove-button-box");
    removeButton.innerText = "Remove";
    removeButton.classList.add("remove-button");
    removeButton.onclick = function() {
        newAlgorithm.remove();
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
    document.getElementById("container").appendChild(newAlgorithm);
}

function logAlgorithmDetails() {
    const algorithms = document.querySelectorAll(".algorithm");
    const algorithmData = [];

    algorithms.forEach(algorithm => {
        const nameInput = algorithm.querySelector(".name-input-box input");
        const colorInput = algorithm.querySelector(".color-input-box input");
        const fileInput = algorithm.querySelector(".choose-file-box input");

        if(nameInput && nameInput.value.trim() !== ""){
            algorithmData.push({
                name: nameInput.value,
                color: colorInput.value,
                file: fileInput.files.length > 0 ? fileInput.files[0].name: "No file selected"
            });
        }
    });

    console.log("Algorithm data:", algorithmData);

    if(algorithmData.length > 0 ){
        sendDataToFlask(algorithmData);
    }
    else{
        alert("Please add at least one algorithm");
    }
}

function sendDataToFlask(algorithmData){
    fetch("/create_graphs", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ algorithms: algorithmData })
    })
    .then(response => response.json())
    .then(data => console.log("Graphs created:", data))
    .catch(error => console.error("Error:", error));
}

// Render the PDF after the page loads
window.onload = function () {
  const pdfUrl = "static/pdf/plot.pdf";
  renderPDF(pdfUrl);
};
