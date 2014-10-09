/* ======= Documentation js modifications ======= */
$(document).ready(function() {
    $('a em').contents().unwrap();
    $('h1').addClass('title');

    /* Jump function. TODO: Test with a domain URL */
    $('a[href*=#]').click(function() {
        if (location.pathname.replace(/^.*\//,'') == this.pathname.replace(/^.*\//,'')
            && location.hostname == this.hostname) {
                var $target = $(this.hash);
                $target = $target.length && $target || $('#' + this.hash.slice(1).replace(/\./g,'\\.'));
                var targetOffset = $target.offset().top - $('header#top').outerHeight(true) - 50;
                $('html,body').animate({scrollTop: targetOffset}, 100);
        }
    });

    /* absolute sidebar handling so it scrolls along with the page at large enough width */
    function setSidebar() {
        if ($(window).width() > 767) {
            var $sphinxsidebar = $('div.sphinxsidebar').removeAttr('style');
            $sphinxsidebar.css('top', document.body.scrollTop);
        }
    }
    setSidebar();
    function resetSidebarPos() {
        if ($(window).width() < 768) {
            var $sphinxsidebar = $('div.sphinxsidebar').removeAttr('style');
            $sphinxsidebar.css('position', 'static');
        }
    }
    resetSidebarPos();

    /* resize & scrolling handlers */
    $(window).scroll(setSidebar);
    $(window).resize(resetSidebarPos);
});
