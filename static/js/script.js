
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


// Render the PDF after the page loads
window.onload = function () {
  const pdfUrl = "static/pdf/plot.pdf";
  renderPDF(pdfUrl);
};
