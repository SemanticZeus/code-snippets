from django.shortcuts import render, get_object_or_404
from .models import Post

def home(request):
    posts = Post.published.all()
    return render(request, 'blog/index.html', {'posts': posts})

def about(request):
    return render(request, 'blog/about.html')

def contact(request):
    return render(request, 'blog/contact.html')

def post_list(request):
    posts = Post.published.all()
    return render(request,
                 'blog/post/list.html',
                 {'posts': posts})


def post_detail(request, id):
    post = get_object_or_404(Post,
                             id=id,
                             status=Post.Status.PUBLISHED)
    return render(request, 'blog/post/test_post.html')
    return render(request,
                  'blog/post/detail.html',
                  {'post': post})
