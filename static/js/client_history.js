console.log("JS LOADED");

const toggle = document.getElementById("toggle-horse-history");

function historyEnabled() {
    return toggle && toggle.checked;
}

document.addEventListener("mouseover", function (e) {

    if (!historyEnabled()) return;

    const cell = e.target.closest(".client-cell, .client-link");
    if (!cell) return;

    const td = cell.closest(".client-cell");
    if (!td) return;

    const clientName = td.dataset.client;
    if (!clientName) return;

    fetch(`/client_history/${encodeURIComponent(clientName)}`)
        .then(r => r.json())
        .then(list => {

            const clean = list.filter(x => x && x.trim());
            const text = clean.length ? clean.join(", ") : "No recent horses";

            const tip = document.getElementById("client-history-tip");
            if (!tip) return;

            const rect = td.getBoundingClientRect();
            tip.innerHTML = text;
            tip.style.top = (rect.top + window.scrollY) + "px";
            tip.style.left = (rect.right + window.scrollX + 4) + "px";
            tip.style.visibility = "visible";
            tip.style.opacity = "1";
        })
        .catch(err => console.error("FETCH ERROR:", err));
});

document.addEventListener("mouseout", function (e) {

    if (!historyEnabled()) return;

    const cell = e.target.closest(".client-cell, .client-link");
    if (!cell) return;

    const tip = document.getElementById("client-history-tip");
    if (!tip) return;

    if (e.relatedTarget && e.relatedTarget.closest("#client-history-tip")) return;
    if (cell.contains(e.relatedTarget)) return;

    tip.style.visibility = "hidden";
    tip.style.opacity = "0";

}, true);

// Optional: hide tooltip instantly when toggle is turned off
toggle.addEventListener("change", () => {
    const tip = document.getElementById("client-history-tip");
    if (!toggle.checked && tip) {
        tip.style.visibility = "hidden";
        tip.style.opacity = "0";
    }
});
