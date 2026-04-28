console.log("JS LOADED");

document.addEventListener("mouseover", function (e) {

    // Match either the TD or the A inside it
    const cell = e.target.closest(".client-cell, .client-link");
    if (!cell) return;

    // Always resolve to the TD that has data-client
    const td = cell.closest(".client-cell");
    if (!td) return;

    const clientName = td.dataset.client;
    if (!clientName) return;

    console.log("HOVER:", clientName);

    fetch(`/client_history/${encodeURIComponent(clientName)}`)
        .then(r => r.json())
        .then(list => {

            console.log("RESULT:", list);

            // Filter out empty strings so tooltip is never blank
            const clean = list.filter(x => x && x.trim());
            const text = clean.length ? clean.join(", ") : "No recent horses";

            const tip = document.getElementById("client-history-tip");
            if (!tip) return;

            tip.innerHTML = text;

            const rect = td.getBoundingClientRect();
            tip.style.top = (rect.top + window.scrollY) + "px";
            tip.style.left = (rect.right + window.scrollX + 4) + "px";

            tip.style.visibility = "visible";
            tip.style.opacity = "1";
        })
        .catch(err => console.error("FETCH ERROR:", err));
});

document.addEventListener("mouseout", function (e) {

    const cell = e.target.closest(".client-cell, .client-link");
    if (!cell) return;

    const tip = document.getElementById("client-history-tip");
    if (!tip) return;

    // If moving into the tooltip, don't hide
    if (e.relatedTarget && e.relatedTarget.closest("#client-history-tip")) return;

    // If still inside the cell, don't hide
    if (cell.contains(e.relatedTarget)) return;

    tip.style.visibility = "hidden";
    tip.style.opacity = "0";

}, true);
