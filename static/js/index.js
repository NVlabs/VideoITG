window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    // Close mobile navbar after clicking an item
    $(".navbar-menu .navbar-item").click(function() {
      if ($(".navbar-burger").hasClass("is-active")) {
        $(".navbar-burger").removeClass("is-active");
        $(".navbar-menu").removeClass("is-active");
      }
    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

    // Back-to-top button
    var backToTop = document.getElementById('backToTop');
    function updateBackToTop() {
      if (!backToTop) return;
      if (window.scrollY > 600) backToTop.classList.add('is-visible');
      else backToTop.classList.remove('is-visible');
    }
    window.addEventListener('scroll', updateBackToTop, { passive: true });
    updateBackToTop();
    if (backToTop) {
      backToTop.addEventListener('click', function() {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    }

    // Image lightbox (click-to-zoom)
    var modal = document.getElementById('imageModal');
    var modalImg = document.getElementById('imageModalImg');
    var modalCaption = document.getElementById('imageModalCaption');
    var prevBtn = document.querySelector('.image-modal-prev');
    var nextBtn = document.querySelector('.image-modal-next');

    var activeGroup = null;
    var activeIndex = -1;

    function getGroupImages(group) {
      if (!group) return [];
      return Array.prototype.slice.call(document.querySelectorAll('img.zoomable[data-zoom-group="' + group + '"]'));
    }

    function openModal(imgEl) {
      if (!modal || !modalImg) return;
      activeGroup = imgEl.getAttribute('data-zoom-group') || null;
      var groupImgs = getGroupImages(activeGroup);
      activeIndex = groupImgs.indexOf(imgEl);

      modalImg.src = imgEl.src;
      var caption = imgEl.getAttribute('data-zoom-caption') || imgEl.alt || '';
      if (modalCaption) modalCaption.textContent = caption;

      modal.classList.add('is-active');
      modal.setAttribute('aria-hidden', 'false');
      document.documentElement.classList.add('is-clipped');

      var showNav = groupImgs.length > 1;
      if (prevBtn) prevBtn.style.display = showNav ? 'block' : 'none';
      if (nextBtn) nextBtn.style.display = showNav ? 'block' : 'none';
    }

    function closeModal() {
      if (!modal) return;
      modal.classList.remove('is-active');
      modal.setAttribute('aria-hidden', 'true');
      document.documentElement.classList.remove('is-clipped');
      activeGroup = null;
      activeIndex = -1;
    }

    function showByDelta(delta) {
      var groupImgs = getGroupImages(activeGroup);
      if (groupImgs.length <= 1) return;
      if (activeIndex < 0) activeIndex = 0;
      activeIndex = (activeIndex + delta + groupImgs.length) % groupImgs.length;
      var imgEl = groupImgs[activeIndex];
      if (!imgEl) return;
      modalImg.src = imgEl.src;
      var caption = imgEl.getAttribute('data-zoom-caption') || imgEl.alt || '';
      if (modalCaption) modalCaption.textContent = caption;
    }

    // Attach click handlers
    var zoomables = document.querySelectorAll('img.zoomable');
    zoomables.forEach(function(img) {
      img.addEventListener('click', function() { openModal(img); });
      img.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' || e.key === ' ') openModal(img);
      });
      img.setAttribute('tabindex', '0');
      img.setAttribute('role', 'button');
      img.setAttribute('aria-label', 'Open image in viewer');
    });

    // Close modal handlers
    document.querySelectorAll('[data-modal-close]').forEach(function(el) {
      el.addEventListener('click', closeModal);
    });

    if (prevBtn) prevBtn.addEventListener('click', function() { showByDelta(-1); });
    if (nextBtn) nextBtn.addEventListener('click', function() { showByDelta(1); });

    document.addEventListener('keydown', function(e) {
      if (!modal || !modal.classList.contains('is-active')) return;
      if (e.key === 'Escape') closeModal();
      if (e.key === 'ArrowLeft') showByDelta(-1);
      if (e.key === 'ArrowRight') showByDelta(1);
    });

})
