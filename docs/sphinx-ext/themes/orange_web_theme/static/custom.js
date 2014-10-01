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

    /* sidebar handling so it is aligned with the last element of the nav-bar */
    function setSidebar() {
        $('div.sphinxsidebar').removeAttr('style');
        if ($(window).width() > 767) {
            var navbar = $('ul.navbar-nav').find('li.nav-item');
            var communityItem = navbar[navbar.length - 1];
            var sidebarRight = $(window).width() - communityItem.offsetLeft - communityItem.offsetWidth + 15;
            $('div.sphinxsidebar').css('right', sidebarRight);

            /* set documentwrapper relative to new sphinxsidebar width */
            var docWrapWidth;
            var docWrapParentWidth = $('div.documentwrapper').parent().width();        
            /* prevent the documentwrapper from being too big */
            (docWrapParentWidth - sidebarRight < $('div.sphinxsidebar').width()) ?
            docWrapWidth = docWrapParentWidth : docWrapWidth = $('div.documentwrapper').parent().width() - sidebarRight;
//             $('div.documentwrapper').css('width', docWrapWidth);
            $('div.documentwrapper').css('width', 'inherit');

            /* if sidebar is too long, return to static */
            if ($(window).height() - $('div.sphinxsidebar').height() - $('header#top').height() < 50 ) {
                $('div.sphinxsidebar').css({'position': 'absolute', 'width': $('div.sphinxsidebar').parent().width()});
            }
        }
    }
    setSidebar();

    /* resize handler */
    function handleResizing() {
        setSidebar();
    }
    $(window).resize(handleResizing);
});
