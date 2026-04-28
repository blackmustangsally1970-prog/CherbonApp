console.log("JS LOADED");

// Event delegation so it ALWAYS works, even with dynamic tables
document.addEventListener("mouseover", function(e) {
    const cell = e.target.closest(".client-cell");
    if (!cell) return;

    const clientName = cell.dataset.client;
    if (!clientName) return;

    console.log("HOVER:", clientName);

    fetch(`/client_history/${encodeURIComponent(clientName)}`)
        .then(r => r.json())
        .then(data => {
            console.log("RESULT:", data);
            // You can add tooltip display here later
        })
        .catch(err => console.error("FETCH ERROR:", err));
});
