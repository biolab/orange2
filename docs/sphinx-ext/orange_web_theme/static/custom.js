/* ======= Documentation js modifications ======= */
jQuery(document).ready(function($) {
    $('a em').contents().unwrap();
    $('h2').first().addClass('title');

    /* Jump function. TODO: Test with a domain URL */
    $('a[href*=#]').click(function() {
        if (location.pathname.replace(/^.*\//,'') == this.pathname.replace(/^.*\//,'')
            && location.hostname == this.hostname) {
                var $target = $(this.hash);
                $target = $target.length && $target || $('#' + this.hash.slice(1).replace(/\./g,'\\.'));
                var targetOffset = $target.offset().top - $('header#top').outerHeight(true) - 80;
                $('html,body').animate({scrollTop: targetOffset}, 100);
        }
    });
});
