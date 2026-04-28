console.log("JS LOADED");

// Event delegation — works even if table is dynamic
document.addEventListener("mouseover", function (e) {
    const cell = e.target.closest(".client-cell");
    if (!cell) return;

    const clientName = cell.dataset.client;
    if (!clientName) return;

    console.log("HOVER:", clientName);

    fetch(`/client_history/${encodeURIComponent(clientName)}`)
        .then(r => r.json())
        .then(data => {
            console.log("RESULT:", data);
            // TODO: Add tooltip display here
        })
        .catch(err => console.error("FETCH ERROR:", err));
});
