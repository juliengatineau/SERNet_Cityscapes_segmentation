function highlightImage(index) {
    // Retirer la classe 'highlight' de toutes les images
    const images = document.querySelectorAll(".image li");
    images.forEach((image) => {
        image.classList.remove("highlight");
    });

    // Ajouter la classe 'highlight' à l'image correspondante
    document.getElementById("image-" + index).classList.add("highlight");

    // Retirer la classe 'active' de tous les éléments de la liste
    const items = document.querySelectorAll(".num_id li");
    items.forEach((item) => {
        item.classList.remove("active");
    });

    // Ajouter la classe 'active' à l'élément de la liste correspondant
    document.getElementById("num-" + index).classList.add("active");
}

// Activer le premier élément au lancement de la page
document.addEventListener("DOMContentLoaded", () => {
    highlightImage(1);
});

function submitForm(idx) {
    // l'image affichée est dans <ul class="image">, items id="image-{{ loop.index }}"
    const img = document.querySelector(`#image-${idx} img`);
    if (!img) {
        alert("Image introuvable");
        return;
    }
    const src = img.getAttribute("src"); // ex: /images/source/xxx_leftImg8bit.png
    const input = document.getElementById("image-url");
    input.value = src;
    document.getElementById("prediction-form").submit();
}

// === Overlay opacity slider (page /display) ===
document.addEventListener("DOMContentLoaded", () => {
    const slider = document.getElementById("alpha");
    const mask = document.getElementById("mask-img");
    const out = document.getElementById("alpha-val");
    if (!slider || !mask) return; // pas sur cette page

    const setOpacity = (v) => {
        const a = Math.max(0, Math.min(100, Number(v) || 0)) / 100;
        mask.style.opacity = String(a);
        if (out) out.textContent = Math.round(a * 100) + "%";
    };

    slider.addEventListener("input", (e) => setOpacity(e.target.value));
    setOpacity(slider.value);
});
