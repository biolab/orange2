/* Decorate numbered list items with images
 *
 */
 
$(function (){
    lists = $('.stamp-list')
    for (i = 0; i < lists.length; i++)
    {
        listitems = lists[i].getElementsByTagName("li")
        for (j = 0; j < listitems.length; j++)
        {
            item = listitems[j];
            url = "url(../_static/images/{0}.png)".replace("{0}", j + 1)
            $(item).css({"list-style-image": url});
        }
    }
});
    