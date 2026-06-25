document.querySelectorAll('.task-check').forEach(cb => {
    cb.addEventListener('change', () => {

        fetch('/update_task', {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: `task_id=${cb.dataset.id}&is_done=${cb.checked}`
        })
        .then(() => {
            // Reload the page so "Done by X" appears
            location.reload();
        });

    });
});
